# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Two explicit toy encoders from resident causal data to the common SL trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tensordict import TensorDict

from embodichain.data_pipeline.datasets import SequenceSpecCfg, TensorDataSource
from embodichain.learning.sl import (
    LossResult,
    SequenceBatchSource,
    SupervisedBatch,
    SupervisedLearner,
    SupervisedTrainer,
    SupervisedTrainerCfg,
)

__all__ = ["main"]


class _MaskedRegression:
    def compute_loss(
        self, model: torch.nn.Module, batch: SupervisedBatch
    ) -> LossResult:
        error = (model(batch.inputs) - batch.targets).square()
        valid = batch.masks["target"].expand_as(error)
        loss = error[valid].mean()
        return LossResult(loss, weight=int(valid.sum()))


def main() -> None:
    """Fit actions or a causal transition without a simulator or data worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("bc", "transition"), default="bc")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    generator = torch.Generator(device="cpu").manual_seed(17)
    drive = torch.randn(8, 17, 1, generator=generator)
    state = torch.zeros(8, 17, 2)
    state[:, 0] = torch.randn(8, 2, generator=generator)
    for time in range(16):
        velocity = state[:, time, 1] + 0.1 * drive[:, time, 0]
        state[:, time + 1] = torch.stack(
            (state[:, time, 0] + 0.1 * velocity, velocity), -1
        )
    schema = {
        "version": "toy-drive-v1",
        "observation": {
            "names": ["position", "velocity"],
            "timing": "pre_action",
            "dtype": "float32",
        },
        "action": {
            "names": ["drive"],
            "semantics": "dimensionless_toy_drive",
            "step_dt": 0.1,
        },
    }
    source = TensorDataSource(
        TensorDict(
            {
                "obs": state,
                "actions": drive,
                "valid": torch.ones(8, 17, dtype=torch.bool),
            },
            [8, 17],
        ),
        schema=schema,
        source_id="synthetic-drive-v1",
    )
    # Real recipes validate their own command/effort/pose, units and frames.
    # This toy action contract is neither robot joint positions nor NeRD efforts.
    if source.schema["action"]["semantics"] != "dimensionless_toy_drive":
        raise ValueError("Toy recipe requires its explicit drive semantics")
    sequence = SequenceSpecCfg(history_length=2, prediction_horizon=2)

    def encode_bc(context: TensorDict) -> SupervisedBatch:
        # Measured history ends at index 1. Actions at 1 and 2 are supervision.
        return SupervisedBatch(
            context["obs"][:, :2].flatten(1),
            context["actions"][:, 1:, 0],
            {"target": context["loss_mask"][:, 1:]},
        )

    def encode_transition(context: TensorDict) -> SupervisedBatch:
        # Future observation 2 is a target only. The current drive is supplied
        # as an input because this recipe predicts its next physical state.
        return SupervisedBatch(
            torch.cat((context["obs"][:, :2].flatten(1), context["actions"][:, 1]), -1),
            context["obs"][:, 2],
            {"target": (context["valid"][:, 1] & context["loss_mask"][:, 2])[:, None]},
        )

    bridge = SequenceBatchSource(
        source,
        sequence,
        encoder=encode_bc if args.encoder == "bc" else encode_transition,
        encoder_version=f"toy-{args.encoder}-v1",
        immutable=True,
    )
    torch.manual_seed(29)
    model = torch.nn.Sequential(
        torch.nn.Linear(4 if args.encoder == "bc" else 5, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 2),
    )
    learner = SupervisedLearner(
        model, _MaskedRegression(), torch.optim.Adam(model.parameters(), lr=0.001)
    )
    trainer = SupervisedTrainer(
        learner,
        bridge,
        SupervisedTrainerCfg(
            total_updates=args.updates,
            batch_size=8,
            output_dir=str(args.output),
            checkpoint_interval=0,
        ),
        signature={
            "recipe": "sequence-source-toy-v1",
            "encoder": args.encoder,
            "action_semantics": schema["action"]["semantics"],
        },
    )
    if args.resume is not None:
        trainer.resume(args.resume)
    print(json.dumps(trainer.fit(), allow_nan=False))


if __name__ == "__main__":
    main()
