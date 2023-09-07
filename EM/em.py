import os
import numpy as np



def em(data, thetas, max_iter=50, eps=1e-3):
    """
    输入:
    data: 观测数据
    thetas:初始化的估计参数值
    max_iter:最大迭代数
    eps:收敛阈值
    输出:
    thetas:估计参数
    """
    # 初始化似然函数值
    ll_old=0
    for i in range(max_iter):
        ### E步: 求隐变量分布
        # 对数似然
        log_like = np.array([np.sum(data*np.log(theta), axis=1) for theta in thetas])
        import pdb;pdb.set_trace()
        # 似然
        like = np.exp(log_like)
        # 求隐变量分布
        ws = like / like.sum(0)
        # 概率加权
        vs = np.array([w[:, None] * data for w in ws])
        ### M步: 更新参数值
        thetas = np.array([v.sum(0)/v.sum() for v in vs])
        # 更新似然函数
        ll_new = np.sum([w*l for w,l in zip(ws, log_like)])
        print(f"Iteration: {i+1}")
        print(f"theta_B = {thetas[0,0]}, theta_C = {thetas[1,0]}, ll = {ll_new}")
        # 满足迭代条件即退出迭代
        if np.abs(ll_new - ll_old) < eps:
            break
        ll_old = ll_new
    return thetas


if __name__ == "__main__":
    # 观测数据，5次独立实验，每次实验10次抛掷的正反面次数
    observed_data = np.array([(5,5),(9,1),(8,2),(4,6),(7,3)])
    thetas = np.array([[0.6,0.4], [0.5,0.5]])
    # EM算法寻优
    thetas = em(observed_data,thetas, max_iter=30, eps=1e-3)
    print(thetas)

