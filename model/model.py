import tensorflow as tf
import os

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

three = tf.Variable(3, dtype=tf.float32)
z = tf.add(3 * x, y, name='z')

model_version = 1
export_path_base = "three_x_plus_y"
export_path = os.path.join(
    tf.compat.as_bytes(export_path_base),
    tf.compat.as_bytes(str(model_version)))
print 'Exporting trained model to', export_path

builder = tf.saved_model.builder.SavedModelBuilder(export_path)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "magic_model": tf.saved_model.signature_def_utils.predict_signature_def(
                inputs={"egg": tf.saved_model.utils.build_tensor_info(x),
                        "bacon": tf.saved_model.utils.build_tensor_info(y)},
                outputs={"spam": tf.saved_model.utils.build_tensor_info(z)})
        })
    builder.save()
