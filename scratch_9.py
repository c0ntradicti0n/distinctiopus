import matplotlib

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from allennlp.commands.elmo import ElmoEmbedder
import spacy
model = 'en_core_web_sm'
nlp = spacy.load(model)

n=200
path = "/home/stefan/PycharmProjects/Sokrates3.1/corpora/edgar allen poe - diddling as exact science.txt"
with open(path) as f:
    lines = f.readlines() [0:40]

text = " ".join(lines)
options_file = './3rdparty/elmo_2x1024_128_2048cnn_1xhighway_options.json'
weight_file = './3rdparty/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
elmo = ElmoEmbedder(options_file=options_file, weight_file=weight_file)

from spacy.tokenizer import Tokenizer
tokenizer = Tokenizer(nlp.vocab)

ts =  nlp(text)


start = next(obj.i for obj in ts if obj.text=="Since" and obj.text)
end = next(obj.i+1 for obj in ts if obj.text=="Jeremys" and obj.i>start)
layer = 2

embeddings = elmo.embed_sentence([x.text for x in ts])
copied_embeddings=np.copy(embeddings)[layer,:,:]

expression_to_look_for = embeddings[:, start:end, :]
print ("Expression: %s "  % str(ts[start:end]))

embeddings = embeddings[layer,:,:]
expression_to_look_for = expression_to_look_for [layer, :, :]


conv = signal.fftconvolve(embeddings, expression_to_look_for, mode='same')
last_axis, last_item = list(enumerate(list(conv.shape)))[-1]
conv = conv.sum(axis=last_axis)
conv = signal.medfilt(conv)

conv_d = np.gradient(conv)

peaks, properties = signal.find_peaks(conv, prominence=2, width=3, height=0, threshold=0.4)
#peaks, properties = signal.find_peaks(-conv_d, prominence=2, width=3, height=0, threshold=0.4)

peaks_width = signal.peak_widths(conv, peaks, rel_height=0.6)

print (conv.shape)


fig, (ax_orig, ax_mag, ax_diff) = plt.subplots(3, 1)

# Plot the filtered image
ax_mag.imshow(embeddings, cmap=plt.cm.gray)
plt.axis('off')

# Plot the filtered image
ax_orig.imshow(copied_embeddings, cmap=plt.cm.gray)
plt.axis('off')

def conv2d(arr):
    return np.array(list(enumerate(arr.tolist())))

def conv3d(arr):
    return np.array(list(enumerate(arr[0].tolist())), )

conv2d(conv)
# Plot the filtered image


plt.plot(conv2d(conv))
plt.plot(peaks,conv[peaks], "x")
plt.axis('off')

plt.plot(conv)
plt.plot(peaks, conv[peaks], "x")
plt.hlines(*peaks_width[1:], color="C2")
plt.show()

for i, v in enumerate(peaks_width[1]):
    l_i = int(round(peaks_width[2][i]))
    r_i = int(round(peaks_width[3][i]))
    print (" ".join([ts[z].text for z in range(l_i,r_i)]))



