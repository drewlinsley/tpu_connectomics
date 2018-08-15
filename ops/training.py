import os
import time
import numpy as np
from datetime import datetime
from utils import logger
from ops import data_utilities
from ops import tf_fun
from db import db


def cv_split(volume, cv_split):
    """Split volume according to cv."""
    vshape = volume.shape
    start = int(vshape[0] * (cv_split[0] / 100.))
    end = int(vshape[0] * (cv_split[1] / 100.))
    return volume[np.arange(start, end)]


def get_z_idx(volume_shape, stride):
    """Get strided z-axis index."""
    z_idx_start = np.arange(
        0,
        volume_shape - stride,
        stride)
    z_idx_end = z_idx_start + stride
    return z_idx_start, z_idx_end


def prepare_idx(z_idx_start, z_idx_end, shuffle, batch_size):
    """Prepare index for training or testing."""
    if shuffle:
        it_idx = np.random.permutation(len(z_idx_start))
    else:
        it_idx = np.arange(len(z_idx_start))
    ss = z_idx_start[it_idx]
    se = z_idx_end[it_idx]

    # Trim and reshape for batches
    batches = np.floor(float(len(ss)) / batch_size).astype(int)
    batch_cut = int(batches * batch_size)
    ss = ss[:batch_cut]
    se = se[:batch_cut]
    ss = ss.reshape(batches, batch_size)
    se = se.reshape(batches, batch_size)
    return ss, se, batches


def sample_data(volume, label, ss, z, dtype, config, fold):
    """Sample from volume and label according to ss."""
    batch_volume = []
    batch_label = []
    for s in ss:
        batch_volume += [volume[s:s + z]]
        batch_label += [label[s:s + z]]
    batch_volume = np.stack(batch_volume, axis=0).astype(dtype)
    batch_label = np.stack(batch_label, axis=0).astype(dtype)
    vshape, lshape = len(batch_volume.shape), len(batch_label.shape)
    if vshape == 4:
        batch_volume = np.expand_dims(batch_volume, axis=-1)
    elif vshape == 5:
        pass
    else:
        raise RuntimeError('Something is wrong with your volume size: %s' % (
            batch_volume.shape))
    if lshape == 4:
        batch_label = np.expand_dims(batch_label, axis=-1)
    elif lshape == 5:
        pass
    else:
        raise RuntimeError('Something is wrong with your label size: %s' % (
            batch_label.shape))
    augmentation_list = getattr(config, '%s_augmentations' % fold)
    if not isinstance(augmentation_list, list):
        augmentation_list = [augmentation_list]
    input_shape = getattr(config, '%s_input_shape' % fold)
    label_shape = getattr(config, '%s_label_shape' % fold)
    if len(augmentation_list):
        batch_volume, batch_label = data_utilities.apply_augmentations(
            volume=batch_volume,
            label=batch_label,
            input_shape=input_shape,
            label_shape=label_shape,
            augmentations=augmentation_list)
    return batch_volume, batch_label


def training_loop(
        config,
        sess,
        summary_op,
        summary_writer,
        saver,
        summary_dir,
        checkpoint_dir,
        prediction_dir,
        train_dict,
        test_dict,
        exp_label,
        train_dataset_module,
        test_dataset_module,
        lr,
        row_id,
        data_structure,
        top_test=5):
    """Run the model training loop."""
    log = logger.get(
        os.path.join(config.log_dir, summary_dir.split(os.path.sep)[-1]))
    step = 0
    train_losses, train_prs, timesteps = (
        [], [], [])
    test_losses, test_prs = (
        [], [])
    # train_losses, train_accs, timesteps, train_arand = (
    #     [], [], [], [])
    # test_losses, test_accs, test_arand = (
    #     [], [], [])

    # Load train and test volumes into memory
    train_data = np.load(train_dataset_module.file_pointer)
    test_data = np.load(test_dataset_module.file_pointer)

    log.info('Loading data.')
    train_volume = train_data['volume']
    train_label = train_data['label']
    test_volume = test_data['volume']
    test_label = test_data['label']

    # Split into CV folds
    train_volume = cv_split(train_volume, config.cross_val['train'])
    train_label = cv_split(train_label, config.cross_val['train'])
    test_volume = cv_split(test_volume, config.cross_val['test'])
    test_label = cv_split(test_label, config.cross_val['test'])

    # Prepare affinity label volumes if requested
    if config.affinity:
        train_label = data_utilities.derive_affinities(
            affinity=config.affinity,
            label_volume=train_label)
        test_label = data_utilities.derive_affinities(
            affinity=config.affinity,
            label_volume=test_label)

    # Derive indices for train/test data
    train_z_idx_start, train_z_idx_end = get_z_idx(
        train_volume.shape[0] - config.train_input_shape[0],
        config.train_stride[-1])
    test_z_idx_start, test_z_idx_end = get_z_idx(
        test_volume.shape[0] - config.test_input_shape[0],
        config.test_stride[-1])

    # Set starting lr
    it_lr = config.lr
    lr_info = None

    # Update DB
    if row_id is not None:
        db.update_results(results=summary_dir, row_id=row_id)

    # Start loop
    em = None
    test_perf = np.ones(top_test) * np.inf
    try:
        for epoch in range(config.epochs):
            # Prepare train idx for current epoch
            log.info('Starting epoch %s/%s' % (epoch, config.epochs))
            train_ss, _, train_batches = prepare_idx(
                train_z_idx_start,
                train_z_idx_end,
                config.shuffle_train,
                config.train_batch_size)
            for idx in range(train_batches):
                # Train batch
                train_batch_volumes, train_batch_labels = sample_data(
                    volume=train_volume,
                    label=train_label,
                    ss=train_ss[idx],
                    z=config.train_input_shape[0],
                    dtype=config.np_dtype,
                    config=config,
                    fold='train')
                start_time = time.time()
                feed_dict = {
                    train_dict['train_images']: train_batch_volumes,
                    train_dict['train_labels']: train_batch_labels,
                    lr: it_lr
                }
                it_train_dict = sess.run(
                    train_dict,
                    feed_dict=feed_dict)
                duration = time.time() - start_time
                train_losses += [it_train_dict['train_loss']]
                train_prs += [it_train_dict['train_pr']]
                # train_accs += [it_train_dict['train_accuracy']]
                # train_arand += [it_train_dict['train_arand']]
                timesteps += [duration]
                try:
                    data_structure.update_training(
                        train_pr=it_train_dict['train_pr'],
                        # train_accuracy=it_train_dict['train_accuracy'],
                        # train_arand=it_train_dict['train_arand'],
                        train_loss=it_train_dict['train_loss'],
                        train_step=step)
                    data_structure.save()
                except Exception as e:
                    log.warning('Failed to update saver class: %s' % e)
                if step % config.test_iters == 0:
                    # Prepare test idx for current epoch
                    test_ss, test_se, test_batches = prepare_idx(
                        test_z_idx_start,
                        test_z_idx_end,
                        config.shuffle_test,
                        config.test_batch_size)
                    it_test_loss = []
                    # it_test_arand = []
                    # it_test_acc = []
                    it_test_scores = []
                    it_test_labels = []
                    it_test_volumes = []
                    it_test_pr = []
                    for num_vals in range(test_batches):
                        log.info('Testing %s...' % num_vals)
                        test_batch_volumes, test_batch_labels = sample_data(
                            volume=test_volume,
                            label=test_label,
                            ss=test_ss[num_vals],
                            z=config.test_input_shape[0],
                            dtype=config.np_dtype,
                            config=config,
                            fold='test')

                        # Test accuracy as the average of n batches
                        feed_dict = {
                            test_dict['test_images']: test_batch_volumes,
                            test_dict['test_labels']: test_batch_labels,
                        }
                        it_test_dict = sess.run(
                            test_dict,
                            feed_dict=feed_dict)
                        # it_test_acc += [it_test_dict['test_accuracy']]
                        # it_test_arand += [it_test_dict['test_arand']]
                        it_test_pr += [it_test_dict['test_pr']]
                        it_test_loss += [it_test_dict['test_loss']]
                        it_test_labels += [it_test_dict['test_labels']]
                        it_test_scores += [it_test_dict['test_logits']]
                        it_test_volumes += [test_batch_volumes]
                    # test_acc = np.mean(it_test_acc)
                    # test_aran = np.mean(it_test_arand)
                    test_lo = np.mean(it_test_loss)
                    test_pr = np.mean(it_test_pr)
                    # test_accs += [test_acc]
                    # test_arand += [test_aran]
                    test_losses += [test_lo]
                    test_prs += [test_pr]

                    # Update data structure
                    try:
                        data_structure.update_test(
                            # test_accuracy=test_acc,
                            # test_arand=test_aran,
                            # test_arand=0.,
                            test_pr=test_pr,
                            test_loss=test_lo,
                            test_step=step,
                            test_lr_info=lr_info,
                            test_lr=it_lr)
                        data_structure.save()
                    except Exception as e:
                        log.warning('Failed to update saver class: %s' % e)

                    # Update data structure
                    try:
                        if row_id is not None:
                            db.update_step(step=step, row_id=row_id)
                    except Exception as e:
                        log.warning('Failed to update step count: %s' % e)

                    # Save checkpoint
                    ckpt_path = os.path.join(
                        checkpoint_dir,
                        'model_%s.ckpt' % step)
                    try:
                        test_check = np.where(test_lo < test_perf)[0]
                        if len(test_check):
                            saver.save(
                                sess,
                                ckpt_path,
                                global_step=step)
                            if len(test_check):
                                test_check = test_check[0]
                            test_perf[test_check] = test_lo
                            log.info('Saved checkpoint to: %s' % ckpt_path)

                            # Save predictions
                            pred_path = os.path.join(
                                prediction_dir,
                                'model_%s' % step)
                            np.savez(
                                pred_path,
                                volumes=it_test_volumes,
                                predictions=it_test_scores,
                                labels=it_test_labels)
                    except Exception as e:
                        log.info('Failed to save checkpoint.')

                    # Update LR
                    it_lr, lr_info = tf_fun.update_lr(
                        it_lr=it_lr,
                        test_losses=test_losses,
                        alg=config.training_routine,
                        lr_info=lr_info)

                    # Summaries
                    # summary_str = sess.run(summary_op)
                    # summary_writer.add_summary(summary_str, step)

                    # Training status and test accuracy
                    format_str = (
                        '%s: step %d, loss = %.2f (%.1f examples/sec; '
                        '%.3f sec/batch) | Training accuracy = %s | '
                        'Test accuracy = %s | Test pr = %s | logdir = %s')
                    log.info(format_str % (
                        datetime.now(),
                        step,
                        it_train_dict['train_loss'],
                        config.train_batch_size / duration,
                        float(duration),
                        # it_train_dict['train_accuracy'],
                        # test_acc,
                        0.,
                        0.,
                        # test_aran,
                        test_pr,
                        summary_dir))
                else:
                    # Training status
                    format_str = (
                        '%s: step %d, loss = %.5f (%.1f examples/sec; '
                        '%.3f sec/batch) | Training accuracy = %s | '
                        'Training pr = %s')
                    log.info(format_str % (
                        datetime.now(),
                        step,
                        it_train_dict['train_loss'],
                        config.train_batch_size / duration,
                        float(duration),
                        # it_train_dict['train_accuracy'],
                        0.,
                        it_train_dict['train_pr']))

                # End iteration
                step += 1
    except Exception as em:
        log.warning('Failed training: %s' % em)
        if row_id is not None:
            db.update_error(error=True, row_id=row_id)

    try:
        data_structure.update_error(msg=em)
        data_structure.save()
    except Exception as e:
        log.warning('Failed to update saver class: %s' % e)
    log.info('Done training for %d epochs, %d steps.' % (config.epochs, step))
    log.info('Saved to: %s' % checkpoint_dir)
    sess.close()

    # Package output variables into a dictionary
    output_dict = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_prs': train_prs,
        'test_prs': test_prs,
        # 'train_accs': train_accs,
        # 'train_arand': train_arand,
        'timesteps': timesteps,
        # 'test_accs': test_accs,
        # 'test_arand': test_arand
    }
    return output_dict
