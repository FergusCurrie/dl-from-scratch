from math import exp, sin, cos 

# https://www.youtube.com/watch?v=lWTSI55HC_c&ab_channel=MachineLearning%26Simulation

# f the target function 
f = lambda x: exp(sin(sin(x)))

# Closed form symbolic derivative of f
f_prime = lambda x: exp(sin(sin(x))) * cos(sin(x)) * cos(x)
print(f(2), f_prime(2))

# Finite difference
print(f'finite difference grad = {(f(2 + 1e-8) - f(2)) / 1e-8}')


def sin_backprop_rule(x):
    y = sin(x)

    # closure funciton
    def sin_pullback(y_cotangent):
        x_cotangent = y_cotangent * cos(x)
        return x_cotangent

    return y, sin_pullback

def exp_backprop_rule(x):
    y = exp(x)

    # closure funciton
    def exp_pullback(y_cotangent):
        x_cotangent = y_cotangent * exp(x)
        return x_cotangent

    return y, exp_pullback

primative_rules = {
    sin: sin_backprop_rule,
    exp: exp_backprop_rule
}

def vjp(chain, primal_point):
    pullback_stack = []
    current_value = primal_point

    # primal pass
    for operation in chain:
        current_value, pullback = primative_rules[operation](current_value)
        pullback_stack.append(pullback)
        pullback_stack.append(pullback)
    
    def pullback(cotangent):
        current_cotangent = cotangent
        for back in reversed(pullback_stack):
            current_cotangent = back(current_cotangent)
        return current_cotangent
    
    return current_value, pullback


res, pullback = vjp([exp, sin, sin], 2)
print(f'autodiff output={res}')
print(f'autodiff grad output={pullback(1)}') # 1.0 because we are evauating vjp - vjp is the effect on input. for scalar->scalar 1.0 is evaluating the derivative there
 

