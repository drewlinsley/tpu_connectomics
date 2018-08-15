import os
import time
import numpy as np
from datetime import datetime
from utils import logger
from db import db
from ops import tf_fun


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
        lr,
        row_id,
        data_structure,
        coord,
        threads,
        top_test=5):
    """Run the model training loop."""
    log = logger.get(
        os.path.join(config.log_dir, summary_dir.split(os.path.sep)[-1]))
    step = 0
    train_losses, train_prs, timesteps = (
        [], [], [])
    test_losses, test_prs = (
        [], [])

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
        while not coord.should_stop():
            start_time = time.time()
            feed_dict = {
                lr: it_lr
            }
            it_train_dict = sess.run(
                train_dict,
                feed_dict=feed_dict)
            import ipdb;ipdb.set_trace()
            duration = time.time() - start_time
            train_losses += [it_train_dict['train_loss']]
            train_prs += [it_train_dict['train_pr']]
            # train_accs += [it_train_dict['train_accuracy']]
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
                it_test_loss = []
                # it_test_arand = []
                # it_test_acc = []
                it_test_scores = []
                it_test_labels = []
                it_test_volumes = []
                it_test_pr = []
                for num_vals in range(config['test_evals']):
                    log.info('Testing %s...' % num_vals)

                    # Test accuracy as the average of n batches
                    it_test_dict = sess.run(test_dict)
                    # it_test_acc += [it_test_dict['test_accuracy']]
                    # it_test_arand += [it_test_dict['test_arand']]
                    it_test_pr += [it_test_dict['test_pr']]
                    it_test_loss += [it_test_dict['test_loss']]
                    it_test_labels += [it_test_dict['test_labels']]
                    it_test_scores += [it_test_dict['test_logits']]
                    it_test_volumes += [it_test_dict['test_images']]
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
                    config.test_batch_size / duration,
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
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

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
