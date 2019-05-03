#!/usr/bin/python3

import pred_anal
import time
#import syslog
#import sys, os
#import subprocess as sp
#import glob


#np.set_printoptions(threshold=np.inf, linewidth=150)




args = parser.parse_args()
if args.verbose_on:
    args.verbose = max(1, args.verbose)

start = time.time()

print('Loading a batch of images...')
image_data = image_generator.flow_from_directory(
    args.data_root,
    target_size=IMAGE_SIZE,
    batch_size=args.batch_size,
    class_mode="sparse")
for image_batch,label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break

print('Classifying a batch of ' +str(args.batch_size) +' images...')
results = model.predict(image_batch)
#print('results shape = ' +str(results.shape))
n_images = results.shape[0]
results_top = np.argmax(results, axis=-1)
K = 5
L = 10
results_topk = pred_anal.result_top_k(results, K)
results_topl = pred_anal.result_top_k(results, L)
labels_batch = imagenet_labels[results_top]
#print(result_top)
#print('Top 1 labels: ' +str(labels_batch))

print('Scoring Top...')
lt = pred_anal.label_types(image_data.class_indices )
accuracy = lt.accuracy( results_top, label_batch)
print("Top 1 accuracy = {0:.3f}".format(accuracy))

accuracy = lt.accuracy( results_topk, label_batch)
print("Top {0} accuracy = {1:.3f}".format(K,accuracy))

accuracy = lt.accuracy( results_topl, label_batch)
print("Top {0} accuracy = {1:.3f}".format(L,accuracy))

end = time.time()
n_images = results.shape[0]
print('Elapsed time to classify ' +str(n_images) +' images: ' +str(end-start))

