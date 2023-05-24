# Fighting_Fake_News

Advances in photo editing and manipulation tools have made it significantly easier to create fake imagery. Learning to detect such manipulations,
however, remains a challenging problem due to the lack of sufficient amounts
of manipulated training data. In this paper, we propose a learning algorithm for
detecting visual image manipulations that is trained only using a large dataset
of real photographs. The algorithm uses the automatically recorded photo EXIF
metadata as supervisory signal for training a model to determine whether an image is self-consistent — that is, whether its content could have been produced
by a single imaging pipeline. We apply this self-consistency model to the task
of detecting and localizing image splices. The proposed method obtains state-ofthe-art performance on several image forensics benchmarks, despite never seeing
any manipulated images at training. That said, it is merely a step in the long quest
for a truly general purpose visual forensics tool.

As an educational project, I successfully implemented the code designed and described in this paper:

https://openaccess.thecvf.com/content_ECCV_2018/papers/Jacob_Huh_Fighting_Fake_News_ECCV_2018_paper.pdf
