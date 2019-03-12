
#
#           Run it on a batch of images
#

batch_size = 20

import time
start = time.time()

print('Loading a batch of images...')
data_root='dogs-vs-cats/train'
#data_root='/home/russell/.keras/datasets/flower_photos'
#data_root='dogs-vs-cats/test'
#data_root='cat_photos'
image_data = image_generator.flow_from_directory(str(data_root),
        target_size=IMAGE_SIZE, batch_size=batch_size)
for image_batch,label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break

print('Classifying a batch of ' +str(batch_size) +' images...')
result_batch = classifier_model.predict(image_batch)
result_top = np.argmax(result_batch, axis=-1)
#K = 5
#result_topk = np.argpartition(result_batch, K)[K:].transpose()
labels_batch = imagenet_labels[result_top]
#print(result_top)
print(labels_batch)

print('Scoring Top1...')
cat_types = np.array([281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293])
cats = np.zeros(batch_size)
for n in range(cats.shape[0]):
    for ct in cat_types:
        if result_top[n] == ct:
            cats[n] = 1

accuracy = np.sum(cats) / float(batch_size)
print("accuracy = " +str(accuracy))


print('Failures:')
for n in range(cats.shape[0]):
    if cats[n] == 0.0:
        print('Example # ' +str(n) +' classified as ' +str(imagenet_labels[result_top[n]]))

end = time.time()
print('Elapsed time to classify ' +str(batch_size) +' images: ' +str(end-start))

