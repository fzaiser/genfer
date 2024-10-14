// Copyright 2018 Paul Scott
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc::c_void;
use std::os::raw::c_char;

pub type Number = f64;
pub type Index = i32;
pub type Int = i32;
pub type Bool = i32;
pub type UserDataPtr = *mut c_void;

//#[repr(C)]
//struct IpoptProblemInfo;
//type IpoptProblem = *mut IpoptProblemInfo;
//type IpoptProblem = *mut c_void; // don't access IpoptProblemInfo, so treat void
pub enum IpoptProblemInfo {} // type-safe way to represent opaque structs
pub type IpoptProblem = *mut IpoptProblemInfo;

#[repr(C)]
#[derive(Debug, PartialEq, Clone)]
#[allow(dead_code)]
pub enum ApplicationReturnStatus {
    SolveSucceeded = 0,
    SolvedToAcceptableLevel = 1,
    InfeasibleProblemDetected = 2,
    SearchDirectionBecomesTooSmall = 3,
    DivergingIterates = 4,
    UserRequestedStop = 5,
    FeasiblePointFound = 6,

    MaximumIterationsExceeded = -1,
    RestorationFailed = -2,
    ErrorInStepComputation = -3,
    MaximumCpuTimeExceeded = -4,
    NotEnoughDegreesOfFreedom = -10,
    InvalidProblemDefinition = -11,
    InvalidOption = -12,
    InvalidNumberDetected = -13,

    UnrecoverableException = -100,
    NonIpoptExceptionThrown = -101,
    InsufficientMemory = -102,
    InternalError = -199,
}

// See IpStdCInterface.h for descriptions
type EvalFCB = extern "C" fn(
    n: Index,
    x: *const Number,
    new_x: Bool,
    obj_value: *mut Number,
    user_data: UserDataPtr,
) -> Bool;

type EvalGradFCB = extern "C" fn(
    n: Index,
    x: *const Number,
    new_x: Bool,
    grad_f: *mut Number,
    user_data: UserDataPtr,
) -> Bool;

type EvalGCB = extern "C" fn(
    n: Index,
    x: *const Number,
    new_x: Bool,
    m: Index,
    g: *mut Number,
    user_data: UserDataPtr,
) -> Bool;

type EvalJacGCB = extern "C" fn(
    n: Index,
    x: *const Number,
    new_x: Bool,
    m: Index,
    nele_jac: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    user_data: UserDataPtr,
) -> Bool;

type EvalHCB = extern "C" fn(
    n: Index,
    x: *const Number,
    new_x: Bool,
    obj_factor: Number,
    m: Index,
    lambda: *const Number,
    new_lambda: Bool,
    nele_hess: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    user_data: UserDataPtr,
) -> Bool;

//type IntermediateCB = extern fn(
//        alg_mod: Index,
//        iter_count: Index,
//        obj_value: Number,
//        inf_pr: Number,
//        inf_du: Number,
//        mu: Number,
//        d_norm: Number,
//        regularization_size: Number,
//        alpha_du: Number,
//        alpha_pr: Number,
//        ls_trials: Number,
//        user_data: UserDataPtr) -> Bool;

#[link(name = "ipopt")]
extern "C" {
    // See IpStdCInterface.h

    pub fn CreateIpoptProblem(
        n: Index,           // number of variables
        x_L: *const Number, // variable lower bounds
        x_U: *const Number, // variable upper bounds
        m: Index,           // number of constraints
        g_L: *const Number, // constraint lower bounds
        g_U: *const Number, // constraint upper bounds
        nele_jac: Index,    // non-zeros in constraint jacobian
        nele_hess: Index,   // non-zeros in lagrangian hessian
        index_style: Index, // indexing style, 0: C, 1 Fortran
        eval_f: EvalFCB,
        eval_g: EvalGCB,
        eval_grad_f: EvalGradFCB,
        eval_jac_g: EvalJacGCB,
        eval_h: EvalHCB,
    ) -> IpoptProblem;

    pub fn FreeIpoptProblem(ipopt_problem: IpoptProblem);

    pub fn AddIpoptStrOption(
        ipopt_problem: IpoptProblem,
        keyword: *const c_char,
        val: *const c_char,
    ) -> Bool;

    pub fn AddIpoptNumOption(
        ipopt_problem: IpoptProblem,
        keyword: *const c_char,
        val: Number,
    ) -> Bool;

    pub fn AddIpoptIntOption(ipopt_problem: IpoptProblem, keyword: *const c_char, val: Int)
        -> Bool;

    //pub fn SetIntermediateCallback(ipopt_problem: IpoptProblem,
    //				     intermediate_cb: IntermediateCB) -> Bool;

    pub fn IpoptSolve(
        ipopt_problem: IpoptProblem,
        x: *mut Number,
        g: *mut Number,
        obj_val: *mut Number,
        mult_g: *mut Number,
        mult_x_L: *mut Number,
        mult_x_U: *mut Number,
        user_data: UserDataPtr,
    ) -> ApplicationReturnStatus;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::ptr;

    #[allow(non_snake_case)]
    extern "C" fn f(
        _n: Index,
        x: *const Number,
        _new_x: Bool,
        obj_value: *mut Number,
        _user_data: UserDataPtr,
    ) -> Bool {
        unsafe {
            *obj_value = (*x) * (*x);
        }
        1
    }

    #[allow(non_snake_case)]
    extern "C" fn f_grad(
        _n: Index,
        _x: *const Number,
        _new_x: Bool,
        grad_f: *mut Number,
        _user_data: UserDataPtr,
    ) -> Bool {
        unsafe {
            *grad_f = 1.0;
        }
        1
    }

    #[allow(non_snake_case)]
    extern "C" fn g(
        _n: Index,
        _x: *const Number,
        _new_x: Bool,
        _m: Index,
        _g: *mut Number,
        _user_data: UserDataPtr,
    ) -> Bool {
        1
    }

    extern "C" fn g_jac(
        _n: Index,
        _x: *const Number,
        _new_x: Bool,
        _m: Index,
        _nele_jac: Index,
        _i_row: *mut Index,
        _j_col: *mut Index,
        values: *mut Number,
        _user_data: UserDataPtr,
    ) -> Bool {
        if values == ptr::null_mut() {
            // set sparsity
        } else {
            // set values
        }
        1
    }

    extern "C" fn l_hess(
        _n: Index,
        _x: *const Number,
        _new_x: Bool,
        _obj_factor: Number,
        _m: Index,
        _lambda: *const Number,
        _new_lambda: Bool,
        _nele_hess: Index,
        i_row: *mut Index,
        j_col: *mut Index,
        values: *mut Number,
        _user_data: UserDataPtr,
    ) -> Bool {
        unsafe {
            if values == ptr::null_mut() {
                *i_row = 0;
                *j_col = 0;
            // set sparsity
            } else {
                *values = 2.0;
                // set values
            }
        }
        1
    }

    #[test]
    fn manual_impl() {
        let x_lb = vec![0.2];
        let x_ub = vec![1.0];
        let g_lb = vec![];
        let g_ub = vec![];
        unsafe {
            let prob = CreateIpoptProblem(
                1,
                x_lb.as_ptr(),
                x_ub.as_ptr(),
                0,
                g_lb.as_ptr(),
                g_ub.as_ptr(),
                0,
                1,
                0,
                f,
                g,
                f_grad,
                g_jac,
                l_hess,
            );
            assert!(prob != ptr::null_mut());
            let opt = CString::new("print_level").unwrap();
            let set = AddIpoptIntOption(prob, opt.as_ptr(), 0);
            assert!(set != 0);
            let opt = CString::new("sb").unwrap();
            let opt_val = CString::new("yes").unwrap();
            AddIpoptStrOption(prob, opt.as_ptr(), opt_val.as_ptr());
            let mut x = vec![0.5];
            let mut obj_val = 0.0;
            let ret = IpoptSolve(
                prob,
                x.as_mut_ptr(),
                ptr::null_mut(),
                &mut obj_val,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            );
            assert_eq!(ret, ApplicationReturnStatus::SolveSucceeded);
            assert_eq!(x[0], 0.2);
            FreeIpoptProblem(prob);
        }
    }
}
