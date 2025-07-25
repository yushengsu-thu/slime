import ray

from slime.ray.placement_group import create_actor_group, create_placement_groups, create_rollout_group
from slime.utils.arguments import parse_args


def add_my_custom_args(parser):
    parser.add_argument(
        "--rollout-num-process",
        type=int,
        default=32,
        help="Number of processes to rollout",
    )
    parser.add_argument(
        "--rollout-num-epoch",
        type=int,
        default=3,
        help="Number of epochs to rollout",
    )
    parser.add_argument(
        "--rollout-input-file",
        type=str,
        default=None,
        help="Input file for rollout",
    )
    return parser


def train(args):
    # allocate the GPUs
    pgs = create_placement_groups(args)

    actor_model = create_actor_group(args, pgs["actor"])

    # create the rollout generator, with sglang engines inside.
    rollout_generator = create_rollout_group(args, pgs["rollout"])

    # sync the initialization (model initalization, load checkpoint, etc.)
    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
    )
    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if args.rollout_global_dataset:
        ray.get(rollout_generator.data_buffer.load.remote(args.start_rollout_id - 1))

    # initialize the connection for weight update during training
    ray.get(actor_model.async_init_weight_update_connections(rollout_generator))

    # always update weight first so that sglang has the loaded weights from training.
    ray.get(actor_model.async_update_weights())

    generation_handles = rollout_generator.async_generate(args.start_rollout_id)
    # async train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):

        ray.get(generation_handles)
        actor_model.get_rollout_data(rollout_id)

        actor_handles = actor_model.async_train(rollout_id, with_data_fetching=False)

        ray.get(actor_handles)
        if (
            args.update_rollout_weights_interval is not None
            and (rollout_id + 1) % args.update_rollout_weights_interval == 0
        ):
            ray.get(actor_model.async_update_weights())

        generation_handles = rollout_generator.async_generate(rollout_id + 1)

        if args.eval_interval is not None and (rollout_id + 1) % args.eval_interval == 0:
            ray.get(rollout_generator.async_generate(rollout_id, evaluation=True))
            ray.get(actor_model.async_eval(rollout_id))

        if args.save_interval is not None and (rollout_id + 1) % args.save_interval == 0:
            ray.get(actor_model.async_save_model(rollout_id))
            if args.rollout_global_dataset:
                ray.get(rollout_generator.data_buffer.save.remote(rollout_id))


if __name__ == "__main__":

    args = parse_args(add_my_custom_args)

    train(args)
