from utils import *

def line_search(model, f, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
    fval = f(True).item()

    for stepfrac in [.5**x for x in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep
        set_flat_params_to(model, x_new)
        fval_new = f(True).item()
        actual_improve = fval - fval_new
        expected_improve = expected_improve_full * stepfrac
        ratio = actual_improve / expected_improve

        if ratio > accept_ratio:
            return True, x_new
    return False, x

def intermediate_line_search(model, f, x, fullstep, expected_improve_full, cc, cost_loss_grad, feasible, max_backtracks=10, accept_ratio=0.1, tol=0.1):
    fval = f(True).item()
    v = cost_loss_grad.dot(fullstep)

    for stepfrac in [.5**x for x in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep
        set_flat_params_to(model, x_new)
        fval_new = f(True).item()
        actual_improve = fval - fval_new
        expected_improve = expected_improve_full * stepfrac
        ratio = actual_improve / expected_improve

        #if feasible check surrogate constraint satisfaction
        if ratio > accept_ratio and (cc + stepfrac*v <= tol or not feasible):
            return True, x_new
    return False, x


def own_line_search(model, f, x, fullstep, expected_improve_full, cc, cost_loss_grad, feasible, max_backtracks=10, accept_ratio=0.1):
    fval = f(True).item()
    v = cost_loss_grad.dot(fullstep)

    if feasible:
        stepfracs = [.5**x for x in range(max_backtracks)]
    else:
        stepfracs = [.5**x for x in [20, 15, 10, 8,4,3,2,1]]

    for stepfrac in stepfracs:
        x_new = x + stepfrac * fullstep
        if feasible:
            set_flat_params_to(model, x_new)
            fval_new = f(True).item()
            actual_improve = fval - fval_new
            expected_improve = expected_improve_full * stepfrac
            ratio = actual_improve / expected_improve

            if ratio > accept_ratio and cc + stepfrac*v <= 0:
                return True, x_new
        else:
            if cc + stepfrac*v <= 0:
                print('stepfrac:', stepfrac)
                return True, x_new
    if feasible:
        return False, x
    else:
        print('stepfrac:', 1)
        return True, x + fullstep