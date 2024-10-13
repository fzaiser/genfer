use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, char, digit1},
    combinator::{cut, eof, map, not, opt, peek, recognize, success, value},
    error::{context, convert_error},
    multi::{many0, many0_count, many1, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, separated_pair, terminated},
    Finish,
};

use crate::ppl::{Comparison, Distribution, Event, Natural, PosRatio, Program, Statement, Var};

type IResult<I, O> = Result<(I, O), nom::Err<nom::error::VerboseError<I>>>;

fn natural(input: &str) -> IResult<&str, Natural> {
    map(delimited(ws, context("natural number", digit1), ws), |n| {
        Natural(n.parse().unwrap())
    })(input)
}

fn u64_natural(input: &str) -> IResult<&str, u64> {
    map(delimited(ws, context("natural number", digit1), ws), |n| {
        n.parse().unwrap()
    })(input)
}

fn natural_list(input: &str) -> IResult<&str, Vec<Natural>> {
    delimited(
        preceded(ws, char('[')),
        cut(context(
            "list of natural numbers",
            separated_list0(char(','), natural),
        )),
        preceded(char(']'), ws),
    )(input)
}

fn pos_ratio(input: &str) -> IResult<&str, PosRatio> {
    delimited(
        ws,
        context(
            "real number",
            alt((
                map(
                    separated_pair(u64_natural, char('/'), cut(u64_natural)),
                    |(n, d)| PosRatio::new(n, d),
                ),
                map(
                    pair(digit1::<&str, _>, opt(preceded(char('.'), cut(digit1)))),
                    |(integer, fractional)| {
                        if let Some(fractional) = fractional {
                            PosRatio::new(
                                (integer.to_owned() + fractional).parse().unwrap(),
                                10u64.checked_pow(fractional.len() as u32).unwrap(),
                            )
                        } else {
                            PosRatio::new(integer.parse().unwrap(), 1)
                        }
                    },
                ),
            )),
        ),
        ws,
    )(input)
}

fn identifier_start(input: &str) -> IResult<&str, &str> {
    alt((alpha1, tag("_")))(input)
}

fn identifier_rest(input: &str) -> IResult<&str, &str> {
    alt((alphanumeric1, tag("_")))(input)
}

fn keyword<'a>(expected: &'static str) -> impl FnMut(&'a str) -> IResult<&'a str, &'a str> {
    terminated(tag(expected), not(peek(identifier_rest)))
}

fn identifier(input: &str) -> IResult<&str, &str> {
    delimited(
        ws,
        context(
            "identifier",
            recognize(pair(identifier_start, cut(many0_count(identifier_rest)))),
        ),
        ws,
    )(input)
}

fn find_var(vars: &[&str], id: &str) -> Option<Var> {
    let var = vars.iter().position(|&v| v == id)?;
    Some(Var(var))
}

fn find_or_create_var<'a>(vars: &mut Vec<&'a str>, id: &'a str) -> Var {
    if let Some(var) = find_var(vars, id) {
        var
    } else {
        vars.push(id);
        Var(vars.len() - 1)
    }
}

fn expect_var(vars: &[&str], id: &str) -> Var {
    find_var(vars, id).unwrap_or_else(|| panic!("Unknown variable {id}"))
}

fn semicolon(input: &str) -> IResult<&str, ()> {
    terminated(ws, char(';'))(input)
}

fn fail(input: &str) -> IResult<&str, Statement> {
    value(Statement::Fail, terminated(keyword("fail"), cut(semicolon)))(input)
}

enum Operand {
    Var(Var),
    Nat(Natural),
}

fn operand<'a>(vars: &[&'a str], input: &'a str) -> IResult<&'a str, Operand> {
    context(
        "comparee",
        alt((
            map(natural, Operand::Nat),
            map(identifier, |id| Operand::Var(expect_var(vars, id))),
        )),
    )(input)
}

fn event_eq(lhs: &Operand, rhs: &Operand) -> Event {
    match (lhs, rhs) {
        (Operand::Var(lhs), Operand::Var(rhs)) => Event::VarComparison(*lhs, Comparison::Eq, *rhs),
        (Operand::Var(var), Operand::Nat(n)) | (Operand::Nat(n), Operand::Var(var)) => {
            Event::InSet(*var, vec![*n])
        }
        (Operand::Nat(lhs), Operand::Nat(rhs)) if lhs == rhs => Event::always(),
        _ => Event::never(),
    }
}

fn event_lt(lhs: &Operand, rhs: &Operand) -> Event {
    match (lhs, rhs) {
        (Operand::Var(lhs), Operand::Var(rhs)) => Event::VarComparison(*lhs, Comparison::Lt, *rhs),
        (Operand::Var(var), Operand::Nat(n)) => Event::InSet(*var, (0..n.0).map(Natural).collect()),
        (Operand::Nat(n), Operand::Var(var)) => {
            Event::InSet(*var, (0..=n.0).map(Natural).collect()).complement()
        }
        (Operand::Nat(lhs), Operand::Nat(rhs)) if lhs.0 < rhs.0 => Event::always(),
        _ => Event::never(),
    }
}

fn event_le(lhs: &Operand, rhs: &Operand) -> Event {
    match (lhs, rhs) {
        (Operand::Var(lhs), Operand::Var(rhs)) => Event::VarComparison(*lhs, Comparison::Le, *rhs),
        (Operand::Var(var), Operand::Nat(n)) => {
            Event::InSet(*var, (0..=n.0).map(Natural).collect())
        }
        (Operand::Nat(n), Operand::Var(var)) => {
            Event::InSet(*var, (0..n.0).map(Natural).collect()).complement()
        }
        (Operand::Nat(lhs), Operand::Nat(rhs)) if lhs.0 <= rhs.0 => Event::always(),
        _ => Event::never(),
    }
}

fn event_in(lhs: &Operand, ns: Vec<Natural>) -> Event {
    match lhs {
        Operand::Var(var) => Event::InSet(*var, ns),
        Operand::Nat(n) if ns.contains(n) => Event::always(),
        Operand::Nat(_) => Event::never(),
    }
}

fn comparison<'a>(vars: &[&'a str], input: &'a str) -> IResult<&'a str, Event> {
    let (input, lhs) = operand(vars, input)?;
    let (input, event) = alt((
        map(
            preceded(char('='), cut(|input| operand(vars, input))),
            |rhs| event_eq(&lhs, &rhs),
        ),
        map(
            preceded(
                alt((tag("<="), tag("≤"))),
                cut(|input| operand(vars, input)),
            ),
            |rhs| event_le(&lhs, &rhs),
        ),
        map(
            preceded(char('<'), cut(|input| operand(vars, input))),
            |rhs| event_lt(&lhs, &rhs),
        ),
        map(
            preceded(alt((keyword("in"), tag("∈"))), cut(natural_list)),
            |ns| event_in(&lhs, ns),
        ),
        map(
            preceded(
                alt((tag("!="), tag("≠"))),
                cut(|input| operand(vars, input)),
            ),
            |rhs| event_eq(&lhs, &rhs).complement(),
        ),
        map(
            preceded(
                alt((tag(">="), tag("≥"))),
                cut(|input| operand(vars, input)),
            ),
            |rhs| event_le(&rhs, &lhs),
        ),
        map(
            preceded(char('>'), cut(|input| operand(vars, input))),
            |rhs| event_lt(&rhs, &lhs),
        ),
        map(
            preceded(alt((keyword("not in"), tag("∉"))), cut(natural_list)),
            |ns| event_in(&lhs, ns).complement(),
        ),
    ))(input)?;
    Ok((input, event))
}

fn data_from_dist(input: &str) -> IResult<&str, Event> {
    let (input, data) = natural(input)?;
    let (input, _) = char('~')(input)?;
    let (input, dist) = cut(|input| distribution(input))(input)?;
    Ok((input, Event::DataFromDist(data, dist)))
}

fn flip(input: &str) -> IResult<&str, Event> {
    let (input, _) = delimited(ws, keyword("flip"), ws)(input)?;
    let (input, p) = delimited(cut(char('(')), pos_ratio, char(')'))(input)?;
    Ok((
        input,
        Event::DataFromDist(Natural(1), Distribution::Bernoulli(p)),
    ))
}

fn atomic_event<'a>(vars: &[&'a str], input: &'a str) -> IResult<&'a str, Event> {
    context(
        "simple event",
        alt((
            |input| {
                map(
                    preceded(
                        alt((tag("!"), keyword("not"))),
                        cut(|input| atomic_event(vars, input)),
                    ),
                    Event::complement,
                )(input)
            },
            |input| {
                delimited(
                    preceded(ws, char('(')),
                    cut(|input| event(vars, input)),
                    preceded(ws, char(')')),
                )(input)
            },
            flip,
            |input| comparison(vars, input),
            data_from_dist,
        )),
    )(input)
}

fn event<'a>(vars: &[&'a str], input: &'a str) -> IResult<&'a str, Event> {
    context("event", |input| {
        let (input, e) = atomic_event(vars, input)?;
        let (input, event) = alt((
            map(
                many1(preceded(
                    pair(ws, alt((keyword("and"), tag("&&")))),
                    cut(|input| event(vars, input)),
                )),
                |mut es| {
                    es.insert(0, e.clone());
                    Event::intersection(es)
                },
            ),
            map(
                many1(preceded(
                    pair(ws, alt((keyword("or"), tag("||")))),
                    cut(|input| event(vars, input)),
                )),
                |mut es| {
                    es.insert(0, e.clone());
                    Event::disjunction(es)
                },
            ),
            success(e.clone()),
        ))(input)?;
        Ok((input, event))
    })(input)
}

fn observe<'a>(vars: &[&'a str], input: &'a str) -> IResult<&'a str, Statement> {
    context(
        "observe statement",
        preceded(
            keyword("observe"),
            cut(|input| {
                let (input, event) = event(vars, input)?;
                let (input, ()) = cut(semicolon)(input)?;
                Ok((
                    input,
                    Statement::IfThenElse {
                        cond: event,
                        then: vec![],
                        els: vec![Statement::Fail],
                    },
                ))
            }),
        ),
    )(input)
}

fn affine_transform<'a>(
    input: &'a str,
    vars: &mut Vec<&'a str>,
    lhs: &'a str,
) -> IResult<&'a str, Statement> {
    let (input, add_previous_value) =
        alt((value(false, tag(":=")), value(true, tag("+="))))(input)?;
    cut(context("assignment", move |input| {
        let (input, (addend, offset)) = alt((
            map(
                pair(
                    pair(opt(terminated(natural, char('*'))), identifier),
                    opt(preceded(char('+'), cut(natural))),
                ),
                |((maybe_factor, w), maybe_offset)| {
                    let factor = maybe_factor.unwrap_or(Natural(1));
                    let offset = maybe_offset.unwrap_or(Natural(0));
                    let var = expect_var(vars, w);
                    (Some((factor, var)), offset)
                },
            ),
            map(natural, |n| (None, n)),
        ))(input)?;
        let lhs = find_or_create_var(vars, lhs);
        let stmt = Statement::Assign {
            var: lhs,
            add_previous_value,
            addend,
            offset,
        };
        Ok((input, stmt))
    }))(input)
}

fn distribution(input: &str) -> IResult<&str, Distribution> {
    let (input, distribution) = identifier(input)?;
    match distribution {
        "Dirac" => map(
            delimited(char('('), cut(natural), char(')')),
            Distribution::Dirac,
        )(input),
        "Bernoulli" => delimited(
            char('('),
            cut(map(pos_ratio, Distribution::Bernoulli)),
            char(')'),
        )(input),
        "Binomial" => delimited(
            char('('),
            cut(map(
                separated_pair(natural, cut(tag(",")), pos_ratio),
                |(n, p)| Distribution::Binomial(n, p),
            )),
            char(')'),
        )(input),
        "Categorical" => delimited(
            char('('),
            cut(context(
                "list of rational numbers",
                map(
                    separated_list1(char(','), pos_ratio),
                    Distribution::Categorical,
                ),
            )),
            char(')'),
        )(input),
        "NegBinomial" => delimited(
            char('('),
            cut(map(
                separated_pair(natural, cut(tag(",")), pos_ratio),
                |(n, p)| Distribution::NegBinomial(n, p),
            )),
            char(')'),
        )(input),
        "Geometric" => map(
            delimited(char('('), cut(pos_ratio), char(')')),
            Distribution::Geometric,
        )(input),
        "Uniform" | "UniformDisc" => delimited(
            char('('),
            cut(map(
                separated_pair(natural, tag(","), natural),
                |(start, end)| Distribution::Uniform { start, end },
            )),
            char(')'),
        )(input),
        _ => panic!("Unknown distribution {distribution}"),
    }
}

fn sample<'a>(
    input: &'a str,
    vars: &mut Vec<&'a str>,
    lhs: &'a str,
) -> IResult<&'a str, Statement> {
    let (input, add_previous_value) =
        alt((value(false, char('~')), value(true, tag("+~"))))(input)?;
    cut(context("distribution", move |input| {
        let var = find_or_create_var(vars, lhs);
        let (input, distribution) = distribution(input)?;
        let stmt = Statement::Sample {
            var,
            distribution,
            add_previous_value,
        };
        Ok((input, stmt))
    }))(input)
}

fn assign<'a>(vars: &mut Vec<&'a str>, input: &'a str) -> IResult<&'a str, Statement> {
    context("assignment", |input| {
        let (input, lhs) = identifier(input)?;
        let (input, stmt) = if input.starts_with('~') || input.starts_with("+~") {
            sample(input, vars, lhs)?
        } else if input.starts_with("-=") {
            let (input, _) = tag("-=")(input)?;
            let (input, offset) = natural(input)?;
            let lhs = find_or_create_var(vars, lhs);
            let stmt = Statement::Decrement { var: lhs, offset };
            (input, stmt)
        } else {
            affine_transform(input, vars, lhs)?
        };
        let (input, ()) = cut(semicolon)(input)?;
        Ok((input, stmt))
    })(input)
}

/// Parses an if-then-else construct of the form:
/// if event {
///    <then>
/// } else {
///   <else>
/// }
fn if_event<'a>(vars: &mut Vec<&'a str>, input: &'a str) -> IResult<&'a str, Statement> {
    let (input, _) = keyword("if")(input)?;
    cut(context("if", |input| {
        let (input, cond) = event(vars, input)?;
        let (input, then) = block(vars, input)?;
        let (input, els) = opt(preceded(
            delimited(ws, keyword("else"), ws),
            cut(|input: &'a str| {
                if keyword("if")(input).is_ok() {
                    let (input, els) = if_event(vars, input)?;
                    Ok((input, vec![els]))
                } else {
                    block(vars, input)
                }
            }),
        ))(input)?;
        let els = els.unwrap_or_default();
        Ok((input, Statement::IfThenElse { cond, then, els }))
    }))(input)
}

fn loop_block<'a>(vars: &mut Vec<&'a str>, input: &'a str) -> IResult<&'a str, Vec<Statement>> {
    let (input, _) = keyword("loop")(input)?;
    cut(context("loop", |input| {
        let (input, iter_count) = natural(input)?;
        let (input, body) = block(vars, input)?;
        let mut stmts = Vec::new();
        for _ in 0..iter_count.0 {
            stmts.extend(body.iter().cloned());
        }
        Ok((input, stmts))
    }))(input)
}

fn while_loop<'a>(vars: &mut Vec<&'a str>, input: &'a str) -> IResult<&'a str, Statement> {
    let (input, _) = keyword("while")(input)?;
    cut(context("while loop", |input| {
        let (input, cond) = event(vars, input)?;
        let (input, unroll) = opt(preceded(preceded(ws, keyword("unroll")), natural))(input)?;
        let unroll = unroll.map(|n| n.0 as usize);
        let (input, body) = block(vars, input)?;
        Ok((input, Statement::While { cond, unroll, body }))
    }))(input)
}

fn ws(mut input: &str) -> IResult<&str, ()> {
    loop {
        input = input.trim_start();
        // Skip comments
        if input.starts_with("#=") {
            let idx = input
                .find("=#")
                .expect("Unterminated comment: found opening `#=` but no closing `=#`")
                + 2;
            input = &input[idx..];
        } else if input.starts_with('#') {
            input = input.trim_start_matches(|c| c != '\n' && c != '\r');
        } else {
            break Ok((input, ()));
        }
    }
}

fn block<'a>(vars: &mut Vec<&'a str>, input: &'a str) -> IResult<&'a str, Vec<Statement>> {
    context(
        "block",
        delimited(
            preceded(ws, char('{')),
            cut(map(many0(|input| statement(vars, input)), |stmts| {
                stmts.into_iter().flatten().collect()
            })),
            cut(preceded(ws, char('}'))),
        ),
    )(input)
}

fn statement<'a>(vars: &mut Vec<&'a str>, input: &'a str) -> IResult<&'a str, Vec<Statement>> {
    context("statement", |input| {
        let (input, ()) = ws(input)?;
        let (input, stmts) = if keyword("if")(input).is_ok() {
            let (input, stmt) = if_event(vars, input)?;
            (input, vec![stmt])
        } else if keyword("observe")(input).is_ok() {
            let (input, stmt) = observe(vars, input)?;
            (input, vec![stmt])
        } else if keyword("loop")(input).is_ok() {
            loop_block(vars, input)?
        } else if keyword("while")(input).is_ok() {
            let (input, stmt) = while_loop(vars, input)?;
            (input, vec![stmt])
        } else if keyword("fail")(input).is_ok() {
            let (input, stmt) = fail(input)?;
            (input, vec![stmt])
        } else {
            let (input, stmt) = assign(vars, input)?;
            (input, vec![stmt])
        };
        let (input, ()) = ws(input)?;
        Ok((input, stmts))
    })(input)
}

fn result<'a>(vars: &[&str], input: &'a str) -> IResult<&'a str, Var> {
    context(
        "return statement",
        map(
            delimited(
                ws,
                delimited(keyword("return"), cut(identifier), opt(tag(";"))),
                ws,
            ),
            |id| expect_var(vars, id),
        ),
    )(input)
}

fn program(input: &str) -> IResult<&str, Program> {
    let mut vars = vec![];
    let (input, stmts) = many0(|input| statement(&mut vars, input))(input)?;
    let stmts = stmts.into_iter().flatten().collect();
    let (input, result) = result(&vars, input)?;
    let (input, _) = preceded(ws, eof)(input)?;
    Ok((input, Program { stmts, result }))
}

pub fn parse_program(input: &str) -> Program {
    match program(input).finish() {
        Ok((_, prog)) => prog,
        Err(e) => {
            panic!("Parse error:\n{}", convert_error(input, e));
        }
    }
}
