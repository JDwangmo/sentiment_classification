### SwDA数据集

### 描述
- 手机通话对话语料
- 开放领域

### 详情
- 这里有两份数据集，分别是：
    - `swb1_dialogact_annot.tar.gz`：解压到([swb1_dialogact_annot](https://github.com/JDwangmo/sentiment_classification/tree/master/dataset/SwDA_26/swb1_dialogact_annot)),由[官方提供](http://web.stanford.edu/~jurafsky/ws97/)，Jurafsky, Daniel, Elizabeth Shriberg, and Debra Biasca. 1997b. Switchboard Dialog Act Corpus. Corpus can be downloaded here as [swb1_dialogact_annot.tar.gz](http://web.stanford.edu/~jurafsky/swb1_dialogact_annot.tar.gz);
    - `swda.zip`：解压到([swda](https://github.com/JDwangmo/sentiment_classification/tree/master/dataset/SwDA_26/swda)),由于上面那份数据集，信息太少，没有元信息(对话或对话者的信息，如主题、性别、教育等等)、树库等（The SwDA is not inherently linked to the Penn Treebank 3 parses of Switchboard, and it is far from straightforward to align the two resources （[Calhoun et al. 2010, §2.4](http://compprag.christopherpotts.net/bibliography.html#NXT-Switchboard)）. In addition, the SwDA is not distributed with the Switchboard's tables of metadata about the conversations and their participants. Christopher Potts等人整理一份包含所有信息的数据集，和封装了代码）。
        - swda.py: 来自 http://compprag.christopherpotts.net/swda.html ，封装了操作类等方便操作，链接中有完整和详细的使用说明。

- swb1_dialogact_annot 数据集 数据解析,以下提到的解析在 swda.zip数据集中基本都已经做好了
    - 每个文件对应一段对话，比如 sw_0001_4325.utt.csv 就对应一段对话，每个文件一开始提供了一些信息，比如对话的时间，如sw_0001_4325.utt.csv中：
        - FILENAME:	4325_1632_1519   ------>   文件名
        - TOPIC#:	323              ------>   主题
        - DATE:		920323           ------>   时间，1992.03.23
    - 话语的标注中，‘+’这个标注比较特殊，表示的是当前句是该用户前面句的补充，即上面用户没说完，被打断，这句话跟前面那句合起来算一句。
        - 比如 sw_0001_4325.utt.csv中,其中一段片段 
            > - o          A.1 utt1: Okay.  /
            > - qw         A.1 utt2: {D So, }   
            > - qy^d       B.2 utt1: [ [ I guess, +   
            > - \+          A.3 utt1: What kind of experience \[ do you, + do you \] have, then with child care? /  
            > - \+          B.4 utt1: I think, ] + {F uh, } I wonder ] if that worked. /  
        -  其中 o qw qy^d + 表示 话语的标注，而其中 + 标注比较特殊，表示跟该用户前一句连接起来是一句完整的话，如“A.3 utt1: What kind of experience \[ do you, + do you \] have, then with child care? /”被标注为“+”，则表示 A用户 的前一句 "A.1 utt2: {D So, } " 要跟当前句连接成一句完整的话，即变成“A  {D So, } What kind of experience \[ do you, + do you \] have, then with child care? ”
        
    - 在话语内容中，'--'这个字符也比较特殊，表示句子不完整，要上一句或者下一句连接起来
        - 比如 sw_0001_4325.utt.csv中,其中一段片段
            - > sv          B.10 utt4: [ I guess + --   
            - > bk          A.11 utt1: Okay. /  
            - > \+          B.12 utt1: -- I guess ] we can start.  {F Uh, } / 
        - 如上：B.10 和 B.12 两句话应该合起来算一句话