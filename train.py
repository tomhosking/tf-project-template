import tensorflow as tf
import numpy as np
import tqdm

# CUDA config
os.environ["CUDA_VISIBLE_DEVICES"]="2"
mem_limit=0.5

# Hyperparameters
tf.app.flags.DEFINE_boolean("train", False, "Training mode?")
tf.app.flags.DEFINE_integer("eval_freq", 100, "Evaluate the model after this many steps")

FLAGS = tf.app.flags.FLAGS

# Data loader
# Should return a tuple of a batch of x,y
def get_batch(epoch, step, dev=False, test=False, size=FLAGS.batch_size):
    pass


def main(_):
    model = TFModel()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options) as sess:
        sess.run(model.init())
        summary_writer = tf.train.SummaryWriter(summary_dir_name, sess.graph)

        x,y,is_training = model.placeholders()
        for e in range(num_epochs):
            for i in tqdm(range(num_steps), desc='Epoch '+str(e)):
                batch_xs, batch_ys = get_batch(e,i)
                _,train_summary = sess.run([model.optimizer(), model.train_summary], feed_dict={x: batch_xs, y: batch_ys, is_training:True})
                summary_writer.add_summary(train_summary, global_step=(e*num_steps+i))

                if i % FLAGS.eval_freq == 0:
                    batch_xs, batch_ys = get_batch(e,i, dev=True)
                    dev_summary = sess.run([model.eval_summary], feed_dict={x: batch_xs, y: batch_ys, is_training:False})
                    summary_writer.add_summary(dev_summary, global_step=(e*num_steps+i))

if __name__ == '__main__':
    tf.app.run()
