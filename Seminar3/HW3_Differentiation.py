
# coding: utf-8

# # Home work 3: Differentiation 

# Since it easy to google every task please please please try to undestand what's going on. The "just answer" thing will be not counted, make sure to present derivation of your solution. It is absolutely OK if you found an answer on web then just exercise in $\LaTeX$ copying it into here.

# Useful links: 
# [1](http://www.machinelearning.ru/wiki/images/2/2a/Matrix-Gauss.pdf)
# [2](http://www.atmos.washington.edu/~dennis/MatrixCalculus.pdf)
# [3](http://cal.cs.illinois.edu/~johannes/research/matrix%20calculus.pdf)
# [4](http://research.microsoft.com/en-us/um/people/cmbishop/prml/index.htm)

# ## ex. 1

# $$  
# y = x^Tx,  \quad x \in \mathbb{R}^N 
# $$

# $$
# \frac{dy}{dx} = 
# $$ 

# In[ ]:




# ## ex. 2

# $$ y = tr(AB) \quad A,B \in \mathbb{R}^{N \times N} $$ 

# $$
# \frac{dy}{dA} =
# $$

# In[ ]:




# ## ex. 3

# $$  
# y = x^TAc , \quad A\in \mathbb{R}^{N \times N}, x\in \mathbb{R}^{N}, c\in \mathbb{R}^{N} 
# $$

# $$
# \frac{dy}{dx} =
# $$

# $$
# \frac{dy}{dA} =
# $$ 

# Hint for the latter (one of the ways): use *ex. 2* result and the fact 
# $$
# tr(ABC) = tr (CAB)
# $$

# In[ ]:




# ## ex. 4

# Classic matrix factorization example. Given matrix $X$ you need to find $A$, $S$ to approximate $X$. This can be done by simple gradient descent iteratively alternating $A$ and $S$ updates.
# $$
# J = || X - AS ||_2^2  , \quad A\in \mathbb{R}^{N \times R} , \quad S\in \mathbb{R}^{R \times M}
# $$
# $$
# \frac{dJ}{dS} = ? 
# $$ 

# ### First approach
# Using ex.2 and the fact:
# $$
# || X ||_2^2 = tr(XX^T) 
# $$ 
# it is easy to derive gradients (you can find it in one of the refs). 

# ### Second approach
# You can use *slightly different techniques* if they suits you. Take a look at this derivation:
# <img src="grad.png">
# (excerpt from [Handbook of blind source separation, Jutten, page 517](https://books.google.ru/books?id=PTbj03bYH6kC&printsec=frontcover&dq=Handbook+of+Blind+Source+Separation&hl=en&sa=X&ved=0ahUKEwi-q_apiJDLAhULvXIKHVXJDWcQ6AEIHDAA#v=onepage&q=Handbook%20of%20Blind%20Source%20Separation&f=false), open for better picture).

# ### Third approach
# And finally we can use chain rule! **YOUR TURN** to do it.
# let $ F = AS $ 
# 
# **Find**
# $$
# \frac{dJ}{dF} =  
# $$ 
# and 
# $$
# \frac{dF}{dS} =  
# $$ 
# (the shape should be $ NM \times RM )$.
# 
# Now it is easy do get desired gradients:
# $$
# \frac{dJ}{dS} =  
# $$ 

# In[ ]:



