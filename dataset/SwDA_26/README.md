### SwDA数据集

### 描述
- 手机通话对话语料
- 开放领域

### 详情
- 这里有两份数据集，分别是：
    - swb1_dialogact_annot.tar.gz：解压到(swb1_dialogact_annot),由[官方提供](http://web.stanford.edu/~jurafsky/ws97/)，Jurafsky, Daniel, Elizabeth Shriberg, and Debra Biasca. 1997b. Switchboard Dialog Act Corpus. Corpus can be downloaded here as [swb1_dialogact_annot.tar.gz](http://web.stanford.edu/~jurafsky/swb1_dialogact_annot.tar.gz);
    - swda.zip：解压到(swda),由于上面那份数据集，信息太少，没有元信息、树库等（The SwDA is not inherently linked to the Penn Treebank 3 parses of Switchboard, and it is far from straightforward to align the two resources （[Calhoun et al. 2010, §2.4](http://compprag.christopherpotts.net/bibliography.html#NXT-Switchboard)）. In addition, the SwDA is not distributed with the Switchboard's tables of metadata about the conversations and their participants. Christopher Potts等人整理一份包含所有信息的数据集，和封装了代码）。
        - swda.py: 来自 http://compprag.christopherpotts.net/swda.html，封装了操作类等方便操作。
