# pyabpoa: abPOA Python interface
## Introduction
pyabpoa provides an easy-to-use interface to [abPOA](https://github.com/yangao07/abPOA), it contains all the APIs that can be used to perform MSA for a set of sequences and consensus calling from the final alignment graph.

## Installation

### Install pyabpoa with pip

pyabpoa can be installed with pip:

```
pip install pyabpoa
```

### Install pyabpoa from source
Alternatively, you can install pyabpoa from source:
```
git clone https://github.com/yangao07/abPOA.git
cd abPOA
make install_py
```

## Examples
The following code illustrates how to use pyabpoa.
```
import pyabpoa as pa
a = pa.msa_aligner()
seqs=[
'CCGAAGA',
'CCGAACTCGA',
'CCCGGAAGA',
'CCGAAGA'
]
res=a.msa(seqs, out_cons=True, out_msa=True, out_pog='pog.png') # perform multiple sequence alignment 
                                                                # generate a figure of alignment graph to pog.png

for seq in res.cons_seq:
    print(seq)  # print consensus sequence

res.print_msa() # print row-column multiple sequence alignment in PIR format
```
You can also try the example script provided in the source folder:
```
python ./python/example.py
```


## APIs

### Class pyabpoa.msa_aligner
```
pyabpoa.msa_aligner(aln_mode='g', ...)
```
This constructs a multiple sequence alignment handler of pyabpoa, it accepts the following arguments:

* **aln_mode**: alignment mode. 'g': global, 'l': local, 'e': extension; default: **'g'**
* **match**: match score; default: **2**
* **gap_open1**: first gap opening penalty; default: **4**
* **gap_ext1**: first gap extension penalty; default: **2**
* **gap_open2**: second gap opening penalty; default: **24**
* **gap_ext2**: second gap extension penalty; default: **1**
* **extra_b**: first adaptive banding paremeter; set as < 0 to disable adaptive banded DP; default: **10**
* **extra_f**: second adaptive banding paremete; the number of extra bases added on both sites of the band is *b+f\*L*, where *L* is the length of the aligned sequence; default : **0.01**
* **is_diploid**: set as 1 if input is diploid datal default: **0**
* **min_freq**: minimum frequency of each consensus to output for diploid datal default: **0.3**


The `msa_aligner` handler provides one method which performs multiple sequence alignment and takes four arguments:
```
pyabpoa.msa_aligner.msa(seqs, out_cons, out_msa, out_pog=None)
```

* **seqs**: a list variable containing a set of input sequences; **positional**
* **out_cons**: a bool variable to ask pyabpoa to generate consensus sequence; **positional**
* **out_msa**: a bool variable to ask pyabpoa to generate RC-MSA; **positional**
* **out_pog**: name of a file (`.png` or `.pdf`) to store the plot of the final alignment graph; **optional**, default: **None**

### Class pyabpoa.msa_result
```
pyabpoa.msa_result(seq_n, cons_n, cons_len, ...)
```
This class describes the information of the generated consensus sequence and the RC-MSA. The returned result of `pyabpoa.msa_aligner.msa()` is an object of this class that has the following properties:

* **seq_n**: number of input aligned sequences
* **cons_n**: number of generated consensus sequences (generally 1, could be 2 if input is specified as diploid)
* **cons_len**: an array of consensus sequence length(s)
* **cons_seq**: an array of consensus sequence(s)
* **msa_len**: size of each row in the RC-MSA
* **msa_seq**: an array containing `seq_n` strings that demonstrates the RC-MSA, each consisting of one input sequence and several `-` indicating the alignment gaps. 

`pyabpoa.msa_result()` has a function of `print_msa` which prints the RC-MSA to screen.

```
pyabpoa.msa_result().print_msa()
```