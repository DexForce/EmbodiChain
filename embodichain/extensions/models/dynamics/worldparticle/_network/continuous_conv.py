import torch

from open3d.ml.torch.python.layers.neighbor_search import RadiusSearch
from open3d.ml.torch.python import ops
from open3d.ml.torch import classes


class FixedRadiusSearch(torch.nn.Module):
    """Fixed radius search for 3D point clouds.

    This layer computes the neighbors for a fixed radius on a point cloud.

    Example:
      This example shows a neighbor search that returns the indices to the
      found neighbors and the distances.::

        import torch
        import open3d.ml.torch as ml3d

        points = torch.randn([20,3])
        queries = torch.randn([10,3])
        radius = 0.8

        nsearch = ml3d.layers.FixedRadiusSearch(return_distances=True)
        ans = nsearch(points, queries, radius)
        # returns a tuple of neighbors_index, neighbors_row_splits, and neighbors_distance


    Arguments:
      metric: Either L1, L2 or Linf. Default is L2.

      ignore_query_point: If True the points that coincide with the center of
        the search window will be ignored. This excludes the query point if
        'queries' and 'points' are the same point cloud.

      return_distances: If True the distances for each neighbor will be returned.
        If False a zero length Tensor will be returned instead.
    """

    def __init__(
        self,
        metric="L2",
        ignore_query_point=False,
        return_distances=False,
        max_hash_table_size=32 * 2**20,
        index_dtype=torch.int32,
        **kwargs,
    ):
        super().__init__()
        self.metric = metric
        self.ignore_query_point = ignore_query_point
        self.return_distances = return_distances
        self.max_hash_table_size = max_hash_table_size
        assert index_dtype in [torch.int32, torch.int64]
        self.index_dtype = index_dtype

    def forward(
        self,
        points,
        queries,
        radius,
        points_row_splits=None,
        queries_row_splits=None,
        hash_table_size_factor=1 / 64,
        hash_table=None,
        return_hash_table=False,
    ):
        """This function computes the neighbors within a fixed radius for each query point.

        Arguments:

          points: The 3D positions of the input points. It can be a RaggedTensor.

          queries: The 3D positions of the query points. It can be a RaggedTensor.

          radius: A scalar with the neighborhood radius

          points_row_splits: Optional 1D vector with the row splits information
            if points is batched. This vector is [0, num_points] if there is
            only 1 batch item.

          queries_row_splits: Optional 1D vector with the row splits information
            if queries is batched.  This vector is [0, num_queries] if there is
            only 1 batch item.

          hash_table_size_factor: Scalar. The size of the hash table as fraction
            of points.

          hash_table: A precomputed hash table generated with build_spatial_hash_table().
            This input can be used to explicitly force the reuse of a hash table in special
            cases and is usually not needed.
            Note that the hash table must have been generated with the same 'points' array.

        Returns:
          3 Tensors in the following order

          neighbors_index
            The compact list of indices of the neighbors. The corresponding query point
            can be inferred from the 'neighbor_count_row_splits' vector.

          neighbors_row_splits
            The exclusive prefix sum of the neighbor count for the query points including
            the total neighbor count as the last element. The size of this array is the
            number of queries + 1.

          neighbors_distance
            Stores the distance to each neighbor if 'return_distances' is True.
            Note that the distances are squared if metric is L2.
            This is a zero length Tensor if 'return_distances' is False.
        """
        if isinstance(points, classes.RaggedTensor):
            points_row_splits = points.row_splits
            points = points.values
        if isinstance(queries, classes.RaggedTensor):
            queries_row_splits = queries.row_splits
            queries = queries.values

        if points_row_splits is None:
            points_row_splits = torch.LongTensor([0, points.shape[0]])
        if queries_row_splits is None:
            queries_row_splits = torch.LongTensor([0, queries.shape[0]])

        if hash_table is None:
            table = ops.build_spatial_hash_table(
                max_hash_table_size=self.max_hash_table_size,
                points=points,
                radius=radius,
                points_row_splits=points_row_splits,
                hash_table_size_factor=hash_table_size_factor,
            )
        else:
            table = hash_table

        result = ops.fixed_radius_search(
            ignore_query_point=self.ignore_query_point,
            return_distances=self.return_distances,
            metric=self.metric,
            points=points,
            queries=queries,
            radius=radius,
            points_row_splits=points_row_splits,
            queries_row_splits=queries_row_splits,
            hash_table_splits=table.hash_table_splits,
            hash_table_index=table.hash_table_index,
            hash_table_cell_splits=table.hash_table_cell_splits,
            index_dtype=self.index_dtype,
        )

        if return_hash_table:
            return result, table

        return result


class ContinuousConv(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        filters,
        kernel_size,
        activation=None,
        use_bias=True,
        kernel_initializer=lambda x: torch.nn.init.uniform_(x, -0.05, 0.05),
        bias_initializer=torch.nn.init.zeros_,
        align_corners=True,
        coordinate_mapping="ball_to_cube_radial",
        interpolation="linear",
        normalize=False,
        offset=None,
        use_dense_layer_for_center=False,
        dense_kernel_initializer=torch.nn.init.xavier_uniform_,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.align_corners = align_corners
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.normalize = normalize
        self.dense_kernel_initializer = dense_kernel_initializer

        if offset is None:
            offset = torch.zeros(size=(3,), dtype=torch.float32)
        self.register_buffer("offset", offset)

        self.use_dense_layer_for_center = use_dense_layer_for_center
        if self.use_dense_layer_for_center:
            self.dense = torch.nn.Linear(self.in_channels, self.filters, bias=False)
            self.dense_kernel_initializer(self.dense.weight)

        kernel_shape = (*self.kernel_size, self.in_channels, self.filters)
        self.kernel = torch.nn.Parameter(
            data=torch.Tensor(*kernel_shape), requires_grad=True
        )
        self.kernel_initializer(self.kernel)

        if self.use_bias:
            self.bias = torch.nn.Parameter(
                data=torch.Tensor(self.filters), requires_grad=True
            )
            self.bias_initializer(self.bias)

    def forward(
        self,
        inp_features,
        inp_positions,
        out_positions,
        extents,
        inp_importance,
        neighbors_index,
        neighbors_row_splits,
        neighbors_importance,
    ):
        # Clone offset to avoid inplace modification issues during backward pass
        # The Open3D continuous_conv operation may modify the offset buffer in-place
        offset = self.offset.clone()

        # for stats and debugging
        num_pairs = neighbors_index.shape[0]
        self._avg_neighbors = num_pairs / out_positions.shape[0]

        extents_rank2 = extents
        while len(extents_rank2.shape) < 2:
            extents_rank2 = torch.unsqueeze(extents_rank2, dim=-1)

        self._conv_values = {
            "filters": self.kernel,
            "out_positions": out_positions,
            "extents": extents_rank2,
            "offset": offset,
            "inp_positions": inp_positions,
            "inp_features": inp_features,
            "inp_importance": inp_importance,
            "neighbors_index": neighbors_index,
            "neighbors_row_splits": neighbors_row_splits,
            "neighbors_importance": neighbors_importance,
            "align_corners": self.align_corners,
            "coordinate_mapping": self.coordinate_mapping,
            "interpolation": self.interpolation,
            "normalize": self.normalize,
        }

        out_features = ops.continuous_conv(**self._conv_values)

        self._conv_output = out_features

        if self.use_dense_layer_for_center:
            self._dense_output = self.dense(inp_features)
            out_features = out_features + self._dense_output

        if self.use_bias:
            out_features += self.bias
        if not self.activation is None:
            out_features = self.activation(out_features)

        return out_features


class ParticleRadiusResearch:

    def __init__(
        self,
        radius_search_metric="L2",
        radius_search_ignore_query_points=True,
        window_function=None,
    ):
        self.radius_search_metric = radius_search_metric
        self.radius_search_ignore_query_points = radius_search_ignore_query_points
        self.window_function = window_function

        self.fixed_radius_search = FixedRadiusSearch(
            metric=self.radius_search_metric,
            ignore_query_point=self.radius_search_ignore_query_points,
            return_distances=not self.window_function is None,
        )

        self.radius_search = RadiusSearch(
            metric=self.radius_search_metric,
            ignore_query_point=self.radius_search_ignore_query_points,
            return_distances=not self.window_function is None,
            normalize_distances=not self.window_function is None,
        )

    def __call__(
        self,
        inp_positions,
        out_positions,
        extents,
        inp_importance=None,
        fixed_radius_search_hash_table=None,
        user_neighbors_index=None,
        user_neighbors_row_splits=None,
        user_neighbors_importance=None,
        inp_row_splits=None,
        out_row_splits=None,
        return_hash_table=False,
    ):

        if return_hash_table:
            assert (
                len(extents.shape) == 0
            ), "returning hash table only supported for fixed radius search with scalar extent"

        if inp_importance is None:
            inp_importance = torch.empty(
                (0,), dtype=torch.float32, device=inp_positions.device
            )

        return_distances = not self.window_function is None

        if not user_neighbors_index is None and not user_neighbors_row_splits is None:

            if user_neighbors_importance is None:
                neighbors_importance = torch.empty(
                    (0,), dtype=torch.float32, device=inp_positions.device
                )
            else:
                neighbors_importance = user_neighbors_importance

            neighbors_index = user_neighbors_index
            neighbors_row_splits = user_neighbors_row_splits

        else:
            if len(extents.shape) == 0:
                radius = 0.5 * extents
                if return_hash_table:
                    self.nns, hash_table = self.fixed_radius_search(
                        inp_positions,
                        queries=out_positions,
                        radius=radius,
                        points_row_splits=inp_row_splits,
                        queries_row_splits=out_row_splits,
                        hash_table=fixed_radius_search_hash_table,
                        return_hash_table=return_hash_table,
                    )
                else:
                    self.nns = self.fixed_radius_search(
                        inp_positions,
                        queries=out_positions,
                        radius=radius,
                        points_row_splits=inp_row_splits,
                        queries_row_splits=out_row_splits,
                        hash_table=fixed_radius_search_hash_table,
                        return_hash_table=return_hash_table,
                    )
                if return_distances:
                    if self.radius_search_metric == "L2":
                        neighbors_distance_normalized = self.nns.neighbors_distance / (
                            radius * radius
                        )
                    else:  # L1
                        neighbors_distance_normalized = (
                            self.nns.neighbors_distance / radius
                        )

            elif len(extents.shape) == 1:
                radii = 0.5 * extents
                self.nns = self.radius_search(
                    inp_positions,
                    queries=out_positions,
                    radii=radii,
                    points_row_splits=inp_row_splits,
                    queries_row_splits=out_row_splits,
                )

            else:
                raise Exception("extents rank must be 0 or 1")

            if self.window_function is None:
                neighbors_importance = torch.empty((0,), dtype=torch.float32)
            else:
                neighbors_importance = self.window_function(
                    neighbors_distance_normalized
                )

            neighbors_index = self.nns.neighbors_index
            neighbors_row_splits = self.nns.neighbors_row_splits

        if return_hash_table:
            return {
                "inp_importance": inp_importance,
                "neighbors_index": neighbors_index,
                "neighbors_row_splits": neighbors_row_splits,
                "neighbors_importance": neighbors_importance,
            }, hash_table
        else:
            return {
                "inp_importance": inp_importance,
                "neighbors_index": neighbors_index,
                "neighbors_row_splits": neighbors_row_splits,
                "neighbors_importance": neighbors_importance,
            }
