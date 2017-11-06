import tensorflow as tf
import os

x = tf.placeholder(tf.float32, shape=None)
y = tf.placeholder(tf.float32, shape=None)

three = tf.Variable(3, dtype=tf.float32)
z = tf.scalar_mul(three, x) + y

model_version = 1
export_path_base = "three_x_plus_y"
export_path = os.path.join(
    tf.compat.as_bytes(export_path_base),
    tf.compat.as_bytes(str(model_version)))
print 'Exporting trained model to', export_path

builder = tf.saved_model.builder.SavedModelBuilder(export_path)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y)
    tensor_info_z = tf.saved_model.utils.build_tensor_info(z)

    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "magic_model": tf.saved_model.signature_def_utils.predict_signature_def(
                inputs={"egg": tensor_info_x, "bacon": tensor_info_y},
                outputs={"spam": tensor_info_z})
        })
    builder.save()
