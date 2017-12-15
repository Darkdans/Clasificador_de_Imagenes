import tensorflow as tf, sys
image_path = sys.argv[1]
##Lee la imagen del argumento
image_data = tf.gfile.FastGFile(image_path, 'rb').read()
# cARGA LAS ETIQUETAS
label_lines = [line.rstrip() for line
    in tf.gfile.GFile("retrained_labels.txt")]
# CARGA LOS GRAFOS DE LA RED
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
# ESTA PARTE COMINEZA A PREDECIR QUE TIPO DE BANDERA ES
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
    
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
