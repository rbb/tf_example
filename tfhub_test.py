
#
#           Run it on a batch of images
#
batch_size = 200

#np.set_printoptions(threshold=np.inf, linewidth=150)

import time
start = time.time()

print('Loading a batch of images...')
data_root='dogs-vs-cats/train'
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
K = 5
L = 10
result_topk = np.argpartition(result_batch, -K)[:,-K:]
result_topl = np.argpartition(result_batch, -L)[:,-L:]
labels_batch = imagenet_labels[result_top]
#print(result_top)
#print('Top 1 labels: ' +str(labels_batch))

cat_types = np.array([281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293])

print('Scoring Top...')
cats = np.zeros(batch_size)
catsk = np.zeros(batch_size)
catsl = np.zeros(batch_size)
for n in range(cats.shape[0]):
    for ct in cat_types:
        if result_top[n] == ct:
            cats[n] = 1
        if np.any(result_topk[n] == ct):
            catsk[n] = 1
        if np.any(result_topl[n] == ct):
            catsl[n] = 1


accuracy = np.sum(cats) / float(batch_size)
print("accuracy Top 1: " +str(accuracy))

accuracyk = np.sum(catsk) / float(batch_size)
print("accuracy Top K=" +str(K) +": " +str(accuracyk))

accuracyl = np.sum(catsl) / float(batch_size)
print("accuracy Top L=" +str(L) +": " +str(accuracyl))


#print('Failures:')
#for n in range(cats.shape[0]):
#    if cats[n] == 0.0:
#        print('Example # ' +str(n) +' classified as ' +str(imagenet_labels[result_top[n]]))

end = time.time()
print('Elapsed time to classify ' +str(batch_size) +' images: ' +str(end-start))

