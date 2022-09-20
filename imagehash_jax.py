import jax
import jax.numpy as jnp

def binary_array_to_hex(arr):
    """
    Function to make a hex string out of a binary array.
    """
    bit_string = ''.join(str(b) for b in 1 * arr.flatten())
    width = int(jnp.ceil(len(bit_string) / 4))
    return '{:0>{width}x}'.format(int(bit_string, 2), width=width)

def convert_L(image):
  #convert image to greyscale using the ITU-R 601-2 luma transform
  # PIL.Image convert('L') method actually uses Floyd-Steinberg dithering
  return jnp.maximum(jnp.minimum(image[:,:,0] * 0.299 + image[:,:,1] * 0.587 + image[:,:,2] * 0.114, 255), 0).astype("uint8")
  
def phash_jax(image, hash_size=8, highfreq_factor=4):
  img_size = hash_size * highfreq_factor
  image = jax.image.resize(convert_L(image), [img_size, img_size], "lanczos3") #convert to greyscale
  dct = jax.scipy.fft.dct(jax.scipy.fft.dct(image, axis=0), axis=1)
  dctlowfreq = dct[:hash_size, :hash_size]
  med = jnp.median(dctlowfreq)
  diff = dctlowfreq > med
  return diff

def dhash_jax(image, hash_size=8):
  #horizontal distance hash
  image = jax.image.resize(convert_L(image), [hash_size, hash_size + 1], "lanczos3") #convert to greyscale
  diff = image[:, 1:] > image[:, :-1]
  return diff

def hash_dist(h1, h2):
  return jnp.count_nonzero(h1.flatten() != h2.flatten())

batch_phash = jax.vmap(jax.jit(phash_jax))
batch_dhash = jax.vmap(jax.jit(dhash_jax))
