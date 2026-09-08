embodichain.lab.sim.atomic_actions.tracking
===========================================

Typed tracking keeps command projection, versioned feedback routing, and
metric evaluation separate from command transport. Policies can use these
contracts for in-flight recovery checks and terminal acceptance without
mixing joint, base-pose, or whole-body error units.

.. automodule:: embodichain.lab.sim.atomic_actions.tracking
   :members:
   :no-index:

   .. rubric:: Channels and endpoint routes

   .. autosummary::

      JOINT_POSITION_CHANNEL
      BASE_POSE_CHANNEL
      WHOLE_BODY_POSE_CHANNEL
      TrackingChannelId
      TrackingFeedbackAddress
      EndpointTrackingFeedbackAddress
      TrackingFeedbackSourceRef
      TrackingProjectorRef
      EndpointTrackingChannelBinding

   .. rubric:: Typed states and metrics

   .. autosummary::

      TrackingState
      JointPositionTrackingState
      PoseTrackingState
      WholeBodyPoseTrackingState
      TrackingMetricCfg
      JointPositionTrackingMetric
      PoseTrackingMetric
      WholeBodyPoseTrackingMetric

   .. rubric:: Tracking policies and command-aligned targets

   .. autosummary::

      InFlightTrackingPolicy
      FeedbackTerminalAcceptance
      TimedTerminalAcceptance
      TerminalAcceptance
      TrackingPolicy
      TrackingSetpoint
      TrackingFrame
      TimedTrackingSequence

   .. rubric:: Runtime extension points and built-ins

   .. autosummary::

      TrackingFeedbackBatch
      TrackingEvaluation
      TrackingFeedbackProvider
      TrackingCommandProjector
      TrackingMetricEvaluator
      TrackingFeedbackProviderRegistry
      TrackingProjectorRegistry
      TrackingEvaluatorRegistry
      PlanningContextTrackingFeedbackProvider
      JointPositionTrackingProjector
      JointPositionTrackingEvaluator
      PoseTrackingEvaluator
      WholeBodyPoseTrackingEvaluator
      TrackingRuntime
