# coding=utf-8
"""
1. encode(decode): 输入单个PMF, 对所有symbol进行编解码;
2. encode(decode)_channel: 输入以通道为单位的pmf [C,Lp], 对[1,C,...]大小的symbol编码.
3. encode(decode)_element: 输入元素为单位的pmf [Lp,...], 对[...]大小的symbol编码.
4. encode(decode)_latent_table: 查表方式的编解码, 需要用到外部的hyperprior.
"""

import sys
from ac.arithmeticcoding import ArithmeticEncoder, ArithmeticDecoder, SimpleFrequencyTable
import numpy as np
import torch


def encode(pmf, sym, enc: ArithmeticEncoder):
    """Encode all the symbols with one single CDF. 
    :param pmf: [Lp], PMF as float, on CPU!
    :param sym: the symbols to encode, as int16, on CPU
    :param enc: Arithmetic Encoder.
    """
    raise NotImplementedError('Error: DEBUG to do!')
    Lp = pmf.shape[0]
    sym = sym.reshape(-1)
    # Construct CDF
    freqs = SimpleFrequencyTable([0] * Lp)
    pmf = (pmf*65536).astype(np.int32)
    cumFreq = np.cumsum(np.clip(pmf, 1, None))
    cumFreq = np.concatenate([[0], cumFreq])
    # updata frequency
    freqs.cumulative = cumFreq
    freqs.total = cumFreq[-1]
    # write bit stream
    for s in sym:
        enc.write(freqs, s)

    enc.finish()
   
def decode(pmf, feat_shape, dec: ArithmeticDecoder):
    """Decode all the symbols with one single CDF.
    :param pmf: [Lp], PMF as float, on CPU!
    :param feat_shape: The shape of feature.
    :param dec: Arithmetic Decoder
    """
    raise NotImplementedError('Error: DEBUG to do!')
    Lp = pmf.shape[0]
    symbol = np.zeros([np.prod(feat_shape)])
    # Construct CDF
    freqs = SimpleFrequencyTable([0] * Lp)
    pmf = (pmf*65536).astype(np.int32)
    cumFreq = np.cumsum(np.clip(pmf, 1, None))
    cumFreq = np.concatenate([[0], cumFreq])
    # updata frequency
    freqs.cumulative = cumFreq
    freqs.total = cumFreq[-1]

    # Read bit stream
    for idx in range(symbol.size):
        symbol[idx] = dec.read(freqs)

    symbol = symbol.reshape(feat_shape)
    return symbol
    
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

def encode_element(pmf, sym, enc: ArithmeticEncoder):
    """Encode the symbols element-wise.
    :param pmf: [Lp,...], pmf matrix, on CPU!
    :param sym: [...], the symbols to encode, as int16, on CPU
    :param enc: Arithmetic Encoder
    """
    Lp = pmf.shape[0]
    pmf = pmf.reshape(Lp,-1)
    sym = sym.reshape(-1)
    assert pmf.shape[1]==sym.size
    freqs = SimpleFrequencyTable([0] * Lp)
    for i in range(sym.size):
        # print(pmf[i])
        pmf_i = (pmf[:,i]*65536).astype(np.int32)
        cumFreq = np.cumsum(np.clip(pmf_i, 1, None))
        cumFreq = np.concatenate([[0], cumFreq])
        # updata frequency
        freqs.cumulative = cumFreq
        freqs.total = cumFreq[-1]
        # write bit stream
        # print(Lp,i,freqs.get_total())
        enc.write(freqs, sym[i])

def decode_element(pmf, feat_shape, dec: ArithmeticDecoder):
    """Decode the symbols element-wise.
    :param pmf: [C, Lp], pmf matrix, on CPU!
    :param feat_shape: The shape of feature.
    :param dec: Arithmetic Decoder
    """
    Lp = pmf.shape[0]
    pmf = pmf.reshape(Lp,-1)
    symbol = np.zeros([np.prod(feat_shape)])
    assert pmf.shape[1]==symbol.size, 'PMF size:%d, Symbol size:%d'%(pmf.shape[1], symbol.size)
    freqs = SimpleFrequencyTable([0] * Lp)
    for i in range(symbol.size):
        pmf_i = (pmf[:,i]*65536).astype(np.int32)
        cumFreq = np.cumsum(np.clip(pmf_i, 1, None))
        cumFreq = np.concatenate([[0], cumFreq])
        # updata frequency
        freqs.cumulative = cumFreq
        freqs.total = cumFreq[-1]
        symbol[i] = dec.read(freqs)
    symbol = symbol.reshape(feat_shape)
    return symbol

def encode_latent_table(crit, hyperprior, sym: np.array, enc: ArithmeticEncoder, Lp=256):
    """Encode the symbols in format of [N,H,W,C] with CDF table. Used for latent.
    :param crit: Hyper-prior criterion.
    :param hyperprior: [N,Ck,H,W], hyper-prior outout.
    :param sym: [N,C,H,W], the symbols to encode, as int16, on CPU
    :param enc: Arithmetic Encoder
    :param Lp: length of CDF
    """
    
    assert hyperprior.shape[-2:] == sym.shape[-2:], "Shape unequal."
    N, C, H, W = sym.shape
    assert N==1, "N==1 support only now!"
    
    freqs = SimpleFrequencyTable([0] * Lp)
    
    # TLp, NCHW
    cdf_table, indice = crit.cdf(hyperprior, C)
    _, Lp_table = cdf_table.shape
    assert Lp+1==Lp_table, "Lp unequal."
    
    # Dimension transfer, modified
    sym = np.transpose(sym, (0,2,3,1))  # NCHW -> NHWC
    indice = np.transpose(indice, (0,2,3,1))  # NCHW -> NHWC
    
    sym_flatten = sym.reshape(-1)
    indice_flatten = indice.reshape(-1)
    
    for i in range(indice_flatten.shape[0]):
        cdf = cdf_table[indice_flatten[i]]
        # updata frequency
        freqs.cumulative = cdf
        freqs.total = cdf[-1]
        # write bit stream
        enc.write(freqs, sym_flatten[i])
          
def decode_latent_table(crit, hyperprior, feat_shape, dec: ArithmeticDecoder, C=128, Lp=256):
    """Decode the symbols with CDF table. Used for latent.
    :param crit: Hyper-prior criterion.
    :param hyperprior: [N,C*K*3,H,W], hyper-prior outout.
    :param feat_shape: The shape of feature.
    :param dec: Arithmetic Decoder
    :param C: Channel of feature
    :param Lp: length of CDF
    """
    N, _, H, W = hyperprior.shape
    assert N==1, "N==1 support only now!"
    
    freqs = SimpleFrequencyTable([0] * Lp)
    symbol = np.zeros([N * C * np.prod(feat_shape)])
    
    # TLp, NCHW
    cdf_table, indice= crit.cdf(hyperprior, C)
    _, Lp_table = cdf_table.shape
    assert Lp+1==Lp_table, "Lp unequal."
    indice_flatten = indice.reshape(-1)
  
    for i in range(indice_flatten.shape[0]):
        cdf = cdf_table[indice_flatten[i]]
        # updata frequency
        freqs.cumulative = cdf
        freqs.total = cdf[-1]
        # Read bit stream
        symbol[i] = dec.read(freqs)
  
    symbol = symbol.reshape([C] + feat_shape)
    return symbol

def encode_latent(crit, hyperprior, symbols: np.array, enc: ArithmeticEncoder, Lp=256):
    """Encode the symbols in format of [N,H,W,C] with CDF. Used for latent.
    :param crit: Hyper-prior criterion.
    :param hyperprior: [N,Ck,H,W], hyper-prior outout.
    :param symbols: [N,C,H,W], the symbols to encode, as int16, on CPU
    :param enc: Arithmetic Encoder
    :param Lp: length of CDF
    """
    
    assert hyperprior.shape[-2:] == symbols.shape[-2:], "Shape unequal."
    N, C, H, W = symbols.shape
    assert N==1, "N==1 support only now!"
    
    freqs = SimpleFrequencyTable([0] * Lp)
    
    # TLp, NCHW
    cdf_table, indice= crit.cdf(hyperprior, C)
    _, Lp_table = cdf_table.shape
    assert Lp+1==Lp_table, "Lp unequal."
    
    # Dimension transfer, modified
    symbols = np.transpose(symbols, (0,2,3,1))  # NCHW -> NHWC
    indice = np.transpose(indice, (0,2,3,1))  # NCHW -> NHWC
    
    symbols_flatten = symbols.reshape(-1)
    indice_flatten = indice.reshape(-1)
    
    for i in range(indice_flatten.shape[0]):
        cdf = cdf_table[indice_flatten[i]]
        # updata frequency
        freqs.cumulative = cdf
        freqs.total = cdf[-1]
        # write bit stream
        enc.write(freqs, symbols_flatten[i])
          
def decode_latent(crit, hyperprior, feat_shape, dec: ArithmeticDecoder, C=128, Lp=256):
    """Decode the symbols with CDF. Used for latent.
    :param crit: Hyper-prior criterion.
    :param hyperprior: [N,C*K*3,H,W], hyper-prior outout.
    :param feat_shape: The shape of feature.
    :param dec: Arithmetic Decoder
    :param C: Channel of feature
    :param Lp: length of CDF
    """
    N, _, H, W = hyperprior.shape
    assert N==1, "N==1 support only now!"
    
    freqs = SimpleFrequencyTable([0] * Lp)
    symbol = np.zeros([N * C * np.prod(feat_shape)])
    
    # TLp, NCHW
    cdf_table, indice= crit.cdf(hyperprior, C)
    _, Lp_table = cdf_table.shape
    assert Lp+1==Lp_table, "Lp unequal."
    indice_flatten = indice.reshape(-1)
  
    for i in range(indice_flatten.shape[0]):
        cdf = cdf_table[indice_flatten[i]]
        # updata frequency
        freqs.cumulative = cdf
        freqs.total = cdf[-1]
        # Read bit stream
        symbol[i] = dec.read(freqs)
  
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
