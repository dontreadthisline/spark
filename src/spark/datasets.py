import gzip
from .autograd import Tensor
'''
def parse_mnist(image_filename:str, label_filename:str)->tuple[Tensor,Tensor]:
    X:Tensor
    y:Tensor
    with gzip.open(image_filename,"rb") as f: 
        magic = int.from_bytes(f.read(4),'big')
        n = int.from_bytes(f.read(4),'big')
        h = int.from_bytes(f.read(4),'big')
        w = int.from_bytes(f.read(4),'big')
        content = list(f.read())
        X_ = array_api.array(content,dtype=np.uint8).reshape((n,h*w)).astype(np.float32) / 255.0
        X = tensor(X_) 
    with gzip.open(label_filename,"rb") as f:
        magic = int.from_bytes(f.read(4),'big')
        n = int.from_bytes(f.read(4),'big')
        content = list(f.read())
        y_ = np.array(content,dtype=np.uint8).reshape(n)
        y = tensor(y_)
    return (X,y)

'''