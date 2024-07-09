import numpy as np
import pandas as pd
import argparse
from sklearn import datasets
from scipy.linalg import block_diag

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=-1)
    parser.add_argument("--output_path",type=str,default="./out.csv")
    parser.add_argument("--export_targets",action="store_true",default=False)
    
    subparsers=parser.add_subparsers(dest="dataset")
    
    synthetic_parser = subparsers.add_parser("scrucca")
    synthetic_parser.add_argument("-N",type=int,default=200)
    synthetic_parser.add_argument("--pro",type=float,default=0.5)

    celeux_one_parser = subparsers.add_parser("celeux_one")
    celeux_one_parser.add_argument("--scenario", type=int, default=1, choices=[1,2,3,4,5])

    celeux_two_parser=subparsers.add_parser("celeux_two")
    celeux_two_parser.add_argument("--scenario",type=int,default=1,choices=[1,2])

    student_parser = subparsers.add_parser("student")
    student_parser.add_argument("--scenario",type=int,default=1,choices=[1,2,3,4,5])

    return parser.parse_args()

def draw_gmm(n, mus,sigmas,pis):
    assert len(mus)==len(sigmas)
    assert len(pis)==len(sigmas)

    # Draw samples from each distribution

    X=[]
    for k in range(len(mus)):
        X+=[np.random.multivariate_normal(mus[k],sigmas[k], size=(n,))]

    # Draw the true cluster from which to draw
    y=np.random.choice(len(mus),p=pis,size=(n,))

    X=[X[k][i].reshape((1,-1)) for i,k in enumerate(y)]

    return np.concatenate(X,axis=0),y

# Check http://www.numdam.org/item/JSFS_2014__155_2_57_0.pdf
def celeux_one(args):
    def create(p, n, mu):
        # Draw the first five variables according to a balance gaussian mixture
        mu1 = np.ones(5) * mu
        mu2 = -mu1
        mu3 = np.zeros(5)
        cov = np.eye(5)

        good_variables,y=draw_gmm(n,[mu1,mu2,mu3],[cov,cov,cov],np.ones(3)/3)

        noise = np.random.normal(size=(n, p-5))

        return np.concatenate([good_variables, noise], axis=1),y
    if args.scenario !=5:
        p=25
    else:
        p=100
    if args.scenario<3:
        n=30
    else:
        n=300
    if args.scenario==1 or args.scenario==3:
        mu=0.6
    else:
        mu=1.7

    return create(p,n,mu)

def celeux_two(args):
    n=2000
    mu1=np.array([0,0])
    mu2=np.array([4,0])
    mu3=np.array([0,2])
    mu4=np.array([4,2])
    cov=np.eye(2)
    if args.scenario==1:
        pis=np.array([0.2,0.3,0.3,0.2])
    else:
        pis=np.ones(4)/4
    good_variables,y=draw_gmm(n,[mu1,mu2,mu3,mu4],[cov,cov,cov,cov],pis)

    if args.scenario==1:
        X3=3*good_variables[:,0]+np.random.normal(loc=0,scale=0.5**2,size=(n,))
        X4_14=np.random.multivariate_normal(np.linspace(0,4,num=11),np.eye(11),size=(n,))
        bad_variables=np.concatenate([X3.reshape((-1,1)),X4_14],axis=1)
    else:
        b=np.array([[0.5,1],[2,0],[0,3],[-1,2],[2,-4],[0.5,0],[4,0.5],[3,0],[2,1]]).T
        rot_pi_3=np.array([[0.5,-np.sqrt(3)/2],[np.sqrt(3)/2,0.5]])
        rot_pi_6=np.array([[np.sqrt(3)/2,-0.5],[0.5,np.sqrt(3)/2]])
        cov_noise=[np.eye(3),0.5*np.eye(2)]
        cov_noise+=[rot_pi_3.T@np.diag(np.array([1,3]))@rot_pi_3]
        cov_noise+=[rot_pi_6.T@np.diag(np.array([2,6]))@rot_pi_6]
        cov_noise=block_diag(*cov_noise)
        noise=np.random.multivariate_normal(np.zeros(9),cov_noise,size=(n,))
        X3_11=np.array([0,0,0.4,0.8,1.2,1.6,2.0,2.4,2.8])+good_variables@b+noise
        X12_14=np.random.multivariate_normal(np.array([3.2,3.6,4]),np.eye(3),size=(n,))
        bad_variables=np.concatenate([X3_11,X12_14],axis=1)

    return np.concatenate([good_variables,bad_variables],axis=1),y

def student(args):
    def sample_student(n,mu,df=1):
        X=np.random.multivariate_normal(np.zeros(len(mu)),np.eye(len(mu)),size=n)
        u=np.random.chisquare(df,n).reshape((-1,1))
        X=np.sqrt(df/u)*X+np.expand_dims(mu,axis=0)

        return X

    if args.scenario !=5:
        p=25
    else:
        p=100
    if args.scenario<3:
        n=30
    else:
        n=300
    if args.scenario==1 or args.scenario==3:
        mu=0.6
    else:
        mu=1.7

    # Sample the 3 informative clusters
    X_0 = sample_student(n,mu*np.ones(5))
    X_1 = sample_student(n,-mu*np.ones(5))
    X_2 = sample_student(n,np.zeros(5))


    y=np.random.choice(3,size=(n,))
    X=np.array([[X_0,X_1,X_2][y[i]][i,:] for i in range(n)])

    noisy_vars = sample_student(n,np.zeros(p-5))

    return np.concatenate([X,noisy_vars],axis=1), y

def synthetic_dataset(args):
    mu1 = np.array([0, 0])
    mu2 = np.array([3, 3])
    sigma1 = np.array([[1.0, 0.5], [0.5, 1.0]])
    sigma2 = np.array([[1.5, -0.7], [-0.7, 1.5]])
    y = np.random.binomial(1, args.pro, size=args.N)
    X0, X1 = np.random.multivariate_normal(mu1, sigma1, size=args.N), np.random.multivariate_normal(mu2, sigma2, size=args.N)
    X = [a if c else b for (a, b, c) in zip(X0, X1, y)]
    X = np.concatenate(X).reshape(-1, 2)
    feature_3 = X[:, 0] + np.random.normal(size=args.N)
    feature_3 = feature_3.reshape((-1, 1))
    feature_4 = np.random.normal(loc=1.5, scale=2**2, size=args.N)
    feature_4 = feature_4.reshape((-1, 1))
    feature_5 = np.random.normal(loc=2, scale=1, size=args.N)
    feature_5 = feature_5.reshape((-1, 1))
    X = np.concatenate([X, feature_3, feature_4, feature_5], axis=1)
    return X,y

def main():
    print("Retrieving args")
    args=get_args()

    if args.seed!=-1:
        np.random.seed(args.seed)

    print(f"Creating dataset {args.dataset}")
    if args.dataset=="scrucca":
        X,y=synthetic_dataset(args)
    elif args.dataset=="celeux_one":
        X,y=celeux_one(args)
    elif args.dataset=="celeux_two":
        X,y=celeux_two(args)
    else:
        X,y=student(args)

    print("Exporting to file")
    pd.DataFrame(X).to_csv(args.output_path,index=False)
    
    if args.export_targets:
    	target_file=args.output_path.replace(".csv","_targets.csv")
    	pd.DataFrame(y).to_csv(target_file,index=False)

if __name__=='__main__':
    main()
