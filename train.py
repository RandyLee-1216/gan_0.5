import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import discriminator
from discriminator import discriminator
import generator
from generator import generator

# Read the dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("official/mnist/dataset.py")


## ----------------------For Training GAN------------------------ ##
z_dimensions = 100
batch_size = 100
tf.reset_default_graph()

# Feeding input noise to the generator
z_placeholder = tf.placeholder(tf.float32, shape=[None, z_dimensions], name='z_placeholder')

# Feeding input images to the discriminator
x_placeholder = tf.placeholder(tf.float32, shape=[None,28,28,1], name='x_placeholder')

# Gz holds the generated images
Gz = generator(z_placeholder, batch_size, z_dimensions)

# Dx holds discriminator prediction probabilities(real)
Dx = discriminator(x_placeholder)

# Dg holds discriminator prediction probabilities(generated)
Dg = discriminator(Gz, reuse_variables=True)

# Two Loss Functions for discriminator
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))

# Loss function for generator
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))

# Get the varaibles for different network
tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

print([v.name for v in d_vars])
print([v.name for v in g_vars])

# Train the discriminator
d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)


## ---------------------Setting TensorBoard----------------------- ##
# From this point forward, reuse variables
sess = tf.Session()
tf.get_variable_scope().reuse_variables()

tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
tf.summary.image('Generated_images', images_for_tensorboard, 5)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)
print("===========The TensorBoard logdir=", logdir, "===========")


## -------------------Start Training Session---------------------- ##
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Pre-train discriminator
print("=====================start pre-training discriminator======================")
for i in range(300):
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})

    if(i % 100 == 0):
        print("dLossReal:", dLossReal, "dLossFake:", dLossFake)
print("=====================pre-train discriminator finished======================")
        
# Train generator and discriminator together
print("===========================================================================")
print("=============================start training================================")
for i in range(200000):
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

    # Train discriminator on both real and fake images
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})

    # Train generator
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

    if i % 50 == 0:
        # Update TensorBoard with summary statistics
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
        writer.add_summary(summary, i)
    
    if i % 1000 == 0:
        # Save the model every 1000 iteration
        save_path = saver.save(sess, "/tmp/model{}.ckpt".format(i))
        print("Model saved in file: %s" % save_path)
        
        # Save a generated image every 1000 iteration
        z_batch = np.random.normal(0, 1, size=[1, z_dimensions])
        generated_images = generator(z_placeholder, 1, z_dimensions)
        images = sess.run(generated_images, {z_placeholder: z_batch})
        plt.imshow(images[0].reshape([28, 28]), cmap='Greys')
        plt.savefig("./img/image{}.png".format(i))
print("============================training finished===============================")
print("============================================================================")
