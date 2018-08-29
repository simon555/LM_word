# -*- coding: utf-8 -*-
from torchtext import data
from torchtext.data.iterator import Iterator
from torchtext.data.dataset import Dataset
from torchtext.data.batch import Batch
import math
from collections import Counter, OrderedDict
from itertools import chain

from tqdm import tqdm


import io
from tqdm import tqdm


class LanguageModelingDataset(data.Dataset):
    """Defines a dataset for language modeling."""

    def __init__(self, path, text_field, args, newline_eos=True,
                 encoding='utf-8', **kwargs):
        """Create a LanguageModelingDataset given a path and a field.

        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            newline_eos: Whether to add an <eos> token for every newline in the
                data file. Default: True.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        self.TEXT=text_field
        self.args = args
        self.bsz=args.bsz+1
        fields = [('text', text_field)]
        
        self.numberOfTokens=0
        
        with io.open(path, encoding=encoding) as f:
            print('read through the dataset to get the total number of tokens...')
            for line in tqdm(f):
                self.numberOfTokens += len(text_field.preprocess(line))
                if newline_eos:
                    self.numberOfTokens += 1
        
        print('found {} tokens in the dataset'.format(self.numberOfTokens))
        
        pads_to_add = self.numberOfTokens % self.bsz
        
        virtual_size = self.numberOfTokens + pads_to_add
        
        number_of_tokens_per_batch = int(virtual_size / self.bsz)
        
        start_generator_at_token=[ i * number_of_tokens_per_batch for i in range(self.bsz)]
        
        print('build the list of generators to read the data lazyly...', end='')
        self.list_of_generators=[ LazyGen(path, newline_eos=newline_eos, bptt=args.bptt,text_field=text_field, encoding=encoding, start_at_token=index).gen() for index in start_generator_at_token ]
        print('done')
        
        
        #make the batch text generator       
        self.text=self.text_gen()
        
        print('end gen')
        
        toyTextForExample=['foo' for _ in range(self.args.bptt * self.bsz)]
        print('built')
        #toy example used only to initiate the class with the attributes of the text.Dataset class
        self.examples = [data.Example.fromlist([toyTextForExample], fields)]            
                 
        print('calling super')
        super(LanguageModelingDataset, self).__init__(
           self.examples, fields, **kwargs)
        
        print('lazy dataset built')
        
    def text_gen(self):
        while True:
            text=[]
            for lazy_gen in self.list_of_generators :
                text+=lazy_gen.__next__()
            yield(text)
    
    
    def build_vocab(self, **kwargs):
        counter = Counter()        
        for i in tqdm(range(math.ceil(self.numberOfTokens / self.bsz/self.args.bptt))):
            x=self.text.__next__()
            if not self.TEXT.sequential:
                x = [x]
            counter.update(x)
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.TEXT.unk_token, self.TEXT.pad_token, self.TEXT.init_token,
                            self.TEXT.eos_token]
            if tok is not None))
        self.TEXT.vocab = self.TEXT.vocab_cls(counter, specials=specials, **kwargs)
    
    
 
            
            
            
    
class LazyGen():
    def __init__(self, path, newline_eos, bptt, text_field, encoding, start_at_token):
        self.start_at_token=start_at_token
        self.path=path
        self.encoding=encoding
        self.text_field=text_field
        self.bptt=bptt
        self.newline_eos=newline_eos
        
    def gen(self):
        while True:
            output=[]
            current_size=0
            current_token=0
            with io.open(self.path, encoding=self.encoding) as f:
                for line in f:
                    for token in self.text_field.preprocess(line):
                        if current_token>=self.start_at_token:                  
                            output.append(token)
                            current_size += 1
                            
                            if current_size == self.bptt:
                                tmp=output
                                output=[]
                                current_size=0
                                yield(tmp)
                        else:
                            current_token+=1
                            
                    if current_token>=self.start_at_token:
                        if self.newline_eos:
                            output.append(u'<eos>')
                            current_size += 1
                            
                            if current_size == self.bptt:
                                tmp=output
                                output=[]
                                current_size=0
                                yield(tmp)
                    else:
                        current_token+=1
                        
        

class lazy_BPTTIterator(Iterator):
    """Defines an lazy iterator for language modeling tasks that use BPTT.

    Provides contiguous streams of examples together with targets that are
    one timestep further forward, for language modeling training with
    backpropagation through time (BPTT). Expects a Dataset with a single
    example and a single field called 'text' and produces Batches with text and
    target attributes.

    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
        bptt_len: Length of sequences for backpropagation through time.
        sort_key: A key to use for sorting examples in order to batch together
            examples with similar lengths and minimize padding. The sort_key
            provided to the Iterator constructor overrides the sort_key
            attribute of the Dataset, or defers to it if None.
        train: Whether the iterator represents a train set.
        repeat: Whether to repeat the iterator for multiple epochs.
        shuffle: Whether to shuffle examples between epochs.
        sort: Whether to sort examples according to self.sort_key.
            Note that repeat, shuffle, and sort default to train, train, and
            (not train).
        device: Device to create batches on. Use -1 for CPU and None for the
            currently active GPU device.
    """

    def __init__(self, dataset, batch_size, bptt_len, **kwargs):
        self.bptt_len = bptt_len
        self.bsz=batch_size + 1 #to take account of the offset due to the LM
        super(lazy_BPTTIterator, self).__init__(dataset, batch_size, **kwargs)

    
    def __len__(self):
        if self.batch_size_fn is not None:
            raise NotImplementedError
        return math.ceil(self.dataset.numberOfTokens / self.bsz/self.bptt_len)
    
    
    def __iter__(self):
        TEXT = self.dataset.fields['text']                
        TEXT.eos_token = None

        while True:  
            for _ in range(int(self.dataset.numberOfTokens/self.bsz/self.bptt_len)):
                text=self.dataset.text.__next__()
                
                data = TEXT.numericalize(
                    [text], device=self.device)
                data = data.view(self.bsz, -1).t().contiguous()
                dataset = Dataset(examples=self.dataset.examples, fields=[
                    ('text', TEXT), ('target', TEXT)])
                yield Batch.fromvars(
                    dataset, self.bsz, train=self.train,
                    text=data[:,:-1],
                    target=data[:,1:])
            if not self.repeat:
                return

 
