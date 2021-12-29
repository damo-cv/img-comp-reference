# coding=utf-8

import sys
from ac.arithmeticcoding import ArithmeticEncoder, ArithmeticDecoder, SimpleFrequencyTable
import numpy as np
import torch


def encode_channel(pmf, sym, enc: ArithmeticEncoder):
    """Encode the symbols across channel with C CDFs. Used for channel encode.
    :param pmf: [C, Lp], pmf matrix, on CPU!
    :param sym: [1, C, ...], the symbols to encode, as int16, on CPU
    :param enc: Arithmetic Encoder
    """
    C, Lp = pmf.shape
    assert sym.shape[1]==C, "Length of CDF and channel of feature  must equal."
    assert sym.shape[0]==1, "N==1 support only now!"
    sym = sym.reshape(C, -1)
    freqs = SimpleFrequencyTable([0] * Lp)
    for i in range(C):
        # encode(pmf[i], sym[i], enc)
        # pmf -> cdf -> pmf
        pmf_c = (pmf[i]*65536).astype(np.int32)
        cumFreq = np.cumsum(np.clip(pmf_c, 1, None))
        cumFreq = np.concatenate([[0], cumFreq])
        # updata frequency
        freqs.cumulative = cumFreq
        freqs.total = cumFreq[-1]
        # write bit stream
        for s in sym[i]:
            enc.write(freqs, s)
     
def decode_channel(pmf, feat_shape, dec: ArithmeticDecoder):
    """Decode the symbols across channel with C CDFs. Used for channel decode.
    :param pmf: [C, Lp], pmf matrix, on CPU!
    :param feat_shape: The shape of feature.
    :param dec: Arithmetic Decoder
    """
    C, Lp = pmf.shape
    symbol = np.zeros([C] + [np.prod(feat_shape)])
    # symbol = np.zeros([C] + feat_shape)
    freqs = SimpleFrequencyTable([0] * Lp)
    for i in range(C):
        # symbol[i] = decode(pmf[i], feat_shape, dec)
        # pmf -> cdf -> pmf
        pmf_c = (pmf[i]*65536).astype(np.int32)
        cumFreq = np.cumsum(np.clip(pmf_c, 1, None))
        cumFreq = np.concatenate([[0], cumFreq])
        # updata frequency
        freqs.cumulative = cumFreq
        freqs.total = cumFreq[-1]
        # Read bit stream
        for idx in range(symbol.shape[1]):
          symbol[i, idx] = dec.read(freqs)
        
    symbol = symbol.reshape([C] + feat_shape)
    return symbol

def write_int(bitout, values, numbits=32):
    """Writes an unsigned integer of the given bit width to the given stream.
    bitout: The underlying bit output stream.
    value(list): unsigned integer.
    numbits: The given bit width.
    """
    for value in values:
        for i in reversed(range(numbits)):
            bitout.write((value >> i) & 1)    # Big endian
        
def read_int(bitin, num, numbits=32):
    """Read an unsigned integer of the given bit width to the given stream.
    bitin: The underlying bit input stream.
    num: The number of int list. 
    numbits: The given bit width.
    """
    buffer_list = []
    for _ in range(num):
        result = 0
        for _ in range(numbits):
            result = (result << 1) | bitin.read_no_eof()    # Big endian
        buffer_list.append(result)
    return buffer_list
