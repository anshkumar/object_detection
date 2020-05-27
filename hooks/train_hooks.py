# https://github.com/panchgonzalez/tf_object_detection_pruning

import tensorflow as tf

model_pruning = tf.contrib.model_pruning

class ModelPruningHook(tf.train.SessionRunHook):
  """Updates model pruning masks and thresholds during training."""

  def __init__(self, target_sparsity, start_step, end_step):
    """Initializes a `ModelPruningHook`.

    This hooks updates masks to a specified sparsity over a certain number of
    training steps.

    Args:
      target_sparsity: float between 0 and 1 with desired sparsity
      start_step: int step to start pruning
      end_step: int step to end pruning
    """
    tf.logging.info("Create ModelPruningHook.")
    self.pruning_hparams = self._get_pruning_hparams(
      target_sparsity=target_sparsity,
      start_step=start_step,
      end_step=end_step
    )

  def begin(self):
    """Called once before using the session.
    When called, the default graph is the one that will be launched in the
    session.  The hook can modify the graph by adding new operations to it.
    After the `begin()` call the graph will be finalized and the other callbacks
    can not modify the graph anymore. Second call of `begin()` on the same
    graph, should not change the graph.
    """
    self.global_step_tensor = tf.train.get_global_step()
    self.mask_update_op = self._get_mask_update_op()
    tf.global_variables_initializer()

  def after_run(self, run_context, run_values):
    """Called after each call to run().
    The `run_values` argument contains results of requested ops/tensors by
    `before_run()`.
    The `run_context` argument is the same one send to `before_run` call.
    `run_context.request_stop()` can be called to stop the iteration.
    If `session.run()` raises any exceptions then `after_run()` is not called.
    Args:
      run_context: A `SessionRunContext` object.
      run_values: A SessionRunValues object.
    """
    run_context.session.run(self.mask_update_op)

  def _get_mask_update_op(self):
    """Fetches model pruning mask update op."""
    graph = tf.get_default_graph()
    with graph.as_default():
      pruning = model_pruning.Pruning(
        self.pruning_hparams,
        global_step=self.global_step_tensor
      )
      mask_update_op = pruning.conditional_mask_update_op()
      pruning.add_pruning_summaries()
      return mask_update_op

  def _get_pruning_hparams(self,
                           target_sparsity=0.5,
                           start_step=0,
                           end_step=-1):
    """Get pruning hyperparameters with updated values.

    Args:
      target_sparsity: float between 0 and 1 with desired sparsity
      start_step: int step to start pruning
      end_step: int step to end pruning
    """
    pruning_hparams = model_pruning.get_pruning_hparams()

    # Set the target sparsity
    pruning_hparams.target_sparsity = target_sparsity

    # Set begin pruning step
    pruning_hparams.begin_pruning_step = start_step
    pruning_hparams.sparsity_function_begin_step = start_step

    # Set final pruning step
    pruning_hparams.end_pruning_step = end_step
    pruning_hparams.sparsity_function_end_step = end_step

    return pruning_hparams
