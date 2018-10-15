import os
import cv2
import timeit
import numpy as np
import tensorflow as tf

camera = cv2.VideoCapture(0)
FaceClassifier =cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile('tf_files1/retrained_labels.txt')]
cv2.namedWindow("Face Recognition",2)

def grabVideoFeed():
    grabbed, frame = camera.read()
    return frame if grabbed else None

def face_crop(frame):
    sub_face = None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FaceClassifier.detectMultiScale(gray, 1.3, 5)
    minisize = (frame.shape[1], frame.shape[0])
    miniframe = cv2.resize(frame, minisize)
    faces = FaceClassifier.detectMultiScale(miniframe)
    for f in faces:
        x, y, w, h = [v for v in f]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
        # Save just the rectangle faces in SubRecFaces
        sub_face = frame[y:y + h, x:x + w]
    #print sub_face
    return sub_face


def initialSetup():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    start_time = timeit.default_timer()

    # This takes 2-5 seconds to run
    # Unpersists graph from file
    with tf.gfile.FastGFile('tf_files1/retrained_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    print 'Took {} seconds to unpersist the graph'.format(timeit.default_timer() - start_time)

initialSetup()

with tf.Session() as sess:
    start_time = timeit.default_timer()

    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    #print 'Took {} seconds to feed data to graph'.format(timeit.default_timer() - start_time)

    while True:
        frame = grabVideoFeed()
        faces = FaceClassifier.detectMultiScale(frame, 1.3, 5)
        for f in faces:
            x, y, w, h = [v for v in f]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
        if frame is None:
            raise SystemError('Issue grabbing the frame')

        frame = cv2.resize(frame, (299, 299), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Face Recognition', frame)

        # adhere to TS graph input structure
        try:
            sub_face = face_crop(frame)
            sub_face = cv2.resize(sub_face, (299, 299), interpolation=cv2.INTER_CUBIC)
            numpy_frame = np.asarray(sub_face)
            numpy_frame = cv2.normalize(numpy_frame.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
            numpy_final = np.expand_dims(numpy_frame, axis=0)
            start_time = timeit.default_timer()

            # This takes 2-5 seconds as well
            predictions = sess.run(softmax_tensor, {'Mul:0': numpy_final})

            print 'Took {} seconds to perform prediction'.format(timeit.default_timer() - start_time)

            start_time = timeit.default_timer()

            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            #print 'Took {} seconds to sort the predictions'.format(timeit.default_timer() - start_time)

            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))
        except:
            print "No Face"

        #print '********* Session Ended *********'

        if cv2.waitKey(1) & 0xFF == ord('q'):
            sess.close()
            break

camera.release()
cv2.destroyAllWindows()