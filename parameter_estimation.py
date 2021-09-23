

import numpy as np
import torch
from typing import Any
from torch import nn
from torch.utils.data import Dataset


def CustomImageDataset(Dataset):
    def __init__(self):
        pass



class GradientDescent:
    def __init__(self, *args):
        self.args = args
    
    def __repr__(self) -> str:
        return f"Gradient descent class represented by {self.args}"
    
    def __len__(self):
        return len(self.args)

    def SGD(gradient: Any, start: Any, learning_rate: float, n_iter: int):
        vector = start
        for _ in range(n_iter):
            diff = -learning_rate + gradient(vector)
            vector += diff
        return vector
        


def principal_function_chapter5():
    t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
    t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
    t_c = torch.tensor(t_c)
    t_u = torch.tensor(t_u)
    

    def model(t_u, w, b):
        return w * t_u + b

    def loss_fn(t_p, t_C):
        squared_diffs = (t_p - t_C)**2
        return squared_diffs.mean()
    

    w = torch.ones(())
    b = torch.zeros(())
    print(w, b)
    t_p = model(t_u, w, b)
    print(t_p)



    loss = loss_fn(t_p, t_c)
    print(loss)


    x = torch.ones(())
    y = torch.ones(3,1)
    z = torch.ones(1,3)
    a = torch.ones(2, 1, 1)
    print(f"shapes: x: {x.shape}, y: {y.shape}")
    print(f"        z: {z.shape}, a: {a.shape}")
    print("x * y:", (x * y).shape)
    print("y * z:", (y * z).shape)
    print("y * z * a:", (y * z * a).shape)


    delta = 0.1
    loss_rate_of_change_w = \
        (loss_fn(model(t_u, w + delta, b), t_c)  - 
        loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)




def main():
    

    def training_loop(n_epochs, optimizer, model, 
        loss_fn, t_u_train, t_u_val, t_c_train, t_c_val):

        for epoch in range(1, n_epochs + 1):
            t_p_train = model(t_u_train)
            loss_train = loss_fn(t_p_train, t_c_train)
            t_p_val = model(t_u_val)


            loss_val = loss_fn(t_p_val, t_c_val)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()


            if epoch == 1 or epoch % 1000 == 0:
                print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},
                e"f" Validation loss {loss_val.item():.4f}")



    linear_model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(linear_model.parameters(), lr=1e-2)

if __name__ == "__main__":
    # gradient = GradientDescent()
    # print(gradient)
    main()