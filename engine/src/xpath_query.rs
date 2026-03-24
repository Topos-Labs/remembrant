//! Semantic XPath parser and query evaluator.
//!
//! Implements the query language from "Semantic XPath: Structured Agentic Memory
//! Access for Conversational AI" (arXiv:2603.01160). The grammar supports axes
//! (`/` child, `//` descendant), node-type selectors, and predicates including
//! positional, range, semantic similarity, attribute matching, comparison,
//! aggregation, and logical combinators.

use std::collections::HashMap;
use std::fmt;

pub use crate::semantic_tree::{TreeNode, TreeNodeType};

// ---------------------------------------------------------------------------
// AST types
// ---------------------------------------------------------------------------

/// A parsed Semantic XPath query.
#[derive(Debug, Clone)]
pub struct XPathQuery {
    pub steps: Vec<QueryStep>,
}

#[derive(Debug, Clone)]
pub struct QueryStep {
    pub axis: Axis,
    pub node_select: NodeSelect,
    pub predicates: Vec<Predicate>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Axis {
    /// `/` — direct children only
    Child,
    /// `//` — all descendants
    Descendant,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeSelect {
    /// Match a specific node type.
    Type(String),
    /// `*` — match any node type.
    Wildcard,
}

#[derive(Debug, Clone)]
pub enum Predicate {
    /// Positional: `[1]`, `[-1]` (from end).
    Position(i32),
    /// Range: `[1:3]`.
    Range(i32, i32),
    /// Semantic similarity: `[node≈"query text"]` or `[node~"query text"]`.
    Semantic(String),
    /// Exact attribute match: `[agent="claude"]`.
    AttrEquals(String, String),
    /// Comparison: `[confidence>0.8]`.
    Comparison(String, CompOp, String),
    /// Aggregation: `[avg(/ToolCall[node≈"auth"])]`.
    Aggregate(AggOp, Box<XPathQuery>, String),
    /// Logical AND.
    And(Box<Predicate>, Box<Predicate>),
    /// Logical OR.
    Or(Box<Predicate>, Box<Predicate>),
    /// Logical NOT.
    Not(Box<Predicate>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompOp {
    Gt,
    Lt,
    Gte,
    Lte,
    Eq,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggOp {
    Avg,
    Min,
    Max,
    GMean,
}

// ---------------------------------------------------------------------------
// ParseError
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub position: usize,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "parse error at position {}: {}",
            self.position, self.message
        )
    }
}

impl std::error::Error for ParseError {}

// ---------------------------------------------------------------------------
// Parser – recursive descent
// ---------------------------------------------------------------------------

struct Parser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn remaining(&self) -> &'a str {
        &self.input[self.pos..]
    }

    fn peek(&self) -> Option<char> {
        self.remaining().chars().next()
    }

    fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    fn skip_ws(&mut self) {
        while self.peek().is_some_and(|c| c.is_ascii_whitespace()) {
            self.advance(1);
        }
    }

    fn err(&self, msg: impl Into<String>) -> ParseError {
        ParseError {
            message: msg.into(),
            position: self.pos,
        }
    }

    fn expect_char(&mut self, ch: char) -> Result<(), ParseError> {
        if self.peek() == Some(ch) {
            self.advance(ch.len_utf8());
            Ok(())
        } else {
            Err(self.err(format!("expected '{}', found {:?}", ch, self.peek())))
        }
    }

    // ----- top-level -----

    fn parse_query(&mut self) -> Result<XPathQuery, ParseError> {
        let mut steps = Vec::new();
        // A query must start with '/' or '//'
        loop {
            self.skip_ws();
            if self.remaining().is_empty() {
                break;
            }
            // We might be inside a nested aggregate context where we stop at ')'
            if self.peek() == Some(')') || self.peek() == Some(']') {
                break;
            }
            let axis = self.parse_axis()?;
            let node_select = self.parse_node_select()?;
            let predicates = self.parse_predicates()?;
            steps.push(QueryStep {
                axis,
                node_select,
                predicates,
            });
        }
        if steps.is_empty() {
            return Err(self.err("empty query"));
        }
        Ok(XPathQuery { steps })
    }

    fn parse_axis(&mut self) -> Result<Axis, ParseError> {
        if self.remaining().starts_with("//") {
            self.advance(2);
            Ok(Axis::Descendant)
        } else if self.remaining().starts_with('/') {
            self.advance(1);
            Ok(Axis::Child)
        } else {
            Err(self.err("expected '/' or '//'"))
        }
    }

    fn parse_node_select(&mut self) -> Result<NodeSelect, ParseError> {
        self.skip_ws();
        if self.peek() == Some('*') {
            self.advance(1);
            return Ok(NodeSelect::Wildcard);
        }
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_alphanumeric() || c == '_') {
            self.advance(1);
        }
        let name = &self.input[start..self.pos];
        if name.is_empty() {
            return Err(self.err("expected node type or '*'"));
        }
        Ok(NodeSelect::Type(name.to_string()))
    }

    fn parse_predicates(&mut self) -> Result<Vec<Predicate>, ParseError> {
        let mut preds = Vec::new();
        loop {
            self.skip_ws();
            if self.peek() != Some('[') {
                break;
            }
            self.advance(1); // consume '['
            self.skip_ws();
            let pred = self.parse_predicate_expr()?;
            self.skip_ws();
            self.expect_char(']')?;
            preds.push(pred);
        }
        Ok(preds)
    }

    /// Parse a predicate expression (handles `and`/`or` at the top level).
    fn parse_predicate_expr(&mut self) -> Result<Predicate, ParseError> {
        let left = self.parse_predicate_unary()?;
        self.skip_ws();

        // Check for 'and' / 'or'
        if self.remaining().starts_with("and ") || self.remaining().starts_with("and\t") {
            self.advance(3);
            self.skip_ws();
            let right = self.parse_predicate_expr()?;
            return Ok(Predicate::And(Box::new(left), Box::new(right)));
        }
        if self.remaining().starts_with("or ") || self.remaining().starts_with("or\t") {
            self.advance(2);
            self.skip_ws();
            let right = self.parse_predicate_expr()?;
            return Ok(Predicate::Or(Box::new(left), Box::new(right)));
        }

        Ok(left)
    }

    fn parse_predicate_unary(&mut self) -> Result<Predicate, ParseError> {
        self.skip_ws();
        // 'not'
        if self.remaining().starts_with("not ") || self.remaining().starts_with("not\t") {
            self.advance(3);
            self.skip_ws();
            let inner = self.parse_predicate_unary()?;
            return Ok(Predicate::Not(Box::new(inner)));
        }
        self.parse_predicate_atom()
    }

    fn parse_predicate_atom(&mut self) -> Result<Predicate, ParseError> {
        self.skip_ws();

        // Aggregation: avg(...), min(...), max(...), gmean(...)
        if let Some(agg_op) = self.try_parse_agg_op() {
            return self.parse_aggregate(agg_op);
        }

        // Semantic: node≈"..." or node~"..."
        if self.remaining().starts_with("node\u{2248}") || self.remaining().starts_with("node~") {
            return self.parse_semantic();
        }

        // Positional / range: starts with digit or '-'
        if self.peek().is_some_and(|c| c.is_ascii_digit() || c == '-') {
            return self.parse_positional_or_range();
        }

        // Strip optional '@' prefix (standard XPath attribute syntax)
        if self.peek() == Some('@') {
            self.advance(1);
        }

        // Attribute / comparison: IDENT op VALUE
        self.parse_attr_or_comparison()
    }

    fn try_parse_agg_op(&mut self) -> Option<AggOp> {
        let ops = [
            ("gmean(", AggOp::GMean),
            ("avg(", AggOp::Avg),
            ("min(", AggOp::Min),
            ("max(", AggOp::Max),
        ];
        for (prefix, op) in &ops {
            if self.remaining().starts_with(prefix) {
                self.advance(prefix.len());
                return Some(*op);
            }
        }
        None
    }

    fn parse_aggregate(&mut self, op: AggOp) -> Result<Predicate, ParseError> {
        // We already consumed "avg(" etc.
        self.skip_ws();
        let subquery = self.parse_query()?;
        self.skip_ws();
        // The subquery's last step should contain a Semantic predicate.
        // According to the grammar: AggOp '(' Query SemanticExpr ')'
        // In practice the semantic predicate is inside the subquery steps.
        // We extract the semantic string from the last step's predicates.
        let mut semantic_text = String::new();
        if let Some(last_step) = subquery.steps.last() {
            for pred in &last_step.predicates {
                if let Predicate::Semantic(s) = pred {
                    semantic_text = s.clone();
                    break;
                }
            }
        }
        self.expect_char(')')?;
        Ok(Predicate::Aggregate(op, Box::new(subquery), semantic_text))
    }

    fn parse_semantic(&mut self) -> Result<Predicate, ParseError> {
        // Consume "node" (4 bytes)
        self.advance(4);
        // Consume ≈ (3-byte UTF-8) or ~ (1 byte)
        if self.remaining().starts_with('\u{2248}') {
            self.advance('\u{2248}'.len_utf8());
        } else if self.remaining().starts_with('~') {
            self.advance(1);
        } else {
            return Err(self.err("expected '≈' or '~' after 'node'"));
        }
        self.skip_ws();
        let s = self.parse_quoted_string()?;
        Ok(Predicate::Semantic(s))
    }

    fn parse_positional_or_range(&mut self) -> Result<Predicate, ParseError> {
        let n = self.parse_integer()?;
        self.skip_ws();
        if self.peek() == Some(':') {
            self.advance(1);
            self.skip_ws();
            let m = self.parse_integer()?;
            Ok(Predicate::Range(n, m))
        } else {
            Ok(Predicate::Position(n))
        }
    }

    fn parse_integer(&mut self) -> Result<i32, ParseError> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.advance(1);
        }
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.advance(1);
        }
        let s = &self.input[start..self.pos];
        s.parse::<i32>()
            .map_err(|_| self.err(format!("invalid integer: '{}'", s)))
    }

    fn parse_attr_or_comparison(&mut self) -> Result<Predicate, ParseError> {
        let ident = self.parse_ident()?;
        self.skip_ws();
        let op = self.parse_comp_op()?;
        self.skip_ws();
        // Value can be quoted string or bare number
        let value = if self.peek() == Some('"') || self.peek() == Some('\'') {
            self.parse_quoted_string()?
        } else {
            self.parse_bare_value()?
        };

        match op {
            CompOp::Eq => {
                // If value looks like a quoted attr, treat as AttrEquals
                Ok(Predicate::AttrEquals(ident, value))
            }
            _ => Ok(Predicate::Comparison(ident, op, value)),
        }
    }

    fn parse_comp_op(&mut self) -> Result<CompOp, ParseError> {
        if self.remaining().starts_with(">=") {
            self.advance(2);
            Ok(CompOp::Gte)
        } else if self.remaining().starts_with("<=") {
            self.advance(2);
            Ok(CompOp::Lte)
        } else if self.remaining().starts_with('>') {
            self.advance(1);
            Ok(CompOp::Gt)
        } else if self.remaining().starts_with('<') {
            self.advance(1);
            Ok(CompOp::Lt)
        } else if self.remaining().starts_with('=') {
            self.advance(1);
            Ok(CompOp::Eq)
        } else {
            Err(self.err("expected comparison operator"))
        }
    }

    fn parse_ident(&mut self) -> Result<String, ParseError> {
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_alphanumeric() || c == '_') {
            self.advance(1);
        }
        let s = &self.input[start..self.pos];
        if s.is_empty() {
            return Err(self.err("expected identifier"));
        }
        Ok(s.to_string())
    }

    fn parse_quoted_string(&mut self) -> Result<String, ParseError> {
        let quote = self
            .peek()
            .ok_or_else(|| self.err("expected quoted string"))?;
        if quote != '"' && quote != '\'' {
            return Err(self.err("expected '\"' or '\\''"));
        }
        self.advance(1);
        let start = self.pos;
        while self.peek().is_some_and(|c| c != quote) {
            self.advance(self.peek().unwrap().len_utf8());
        }
        let s = self.input[start..self.pos].to_string();
        self.expect_char(quote)?;
        Ok(s)
    }

    fn parse_bare_value(&mut self) -> Result<String, ParseError> {
        let start = self.pos;
        while self
            .peek()
            .is_some_and(|c| !c.is_ascii_whitespace() && c != ']' && c != ')')
        {
            self.advance(1);
        }
        let s = &self.input[start..self.pos];
        if s.is_empty() {
            return Err(self.err("expected value"));
        }
        Ok(s.to_string())
    }
}

/// Parse a Semantic XPath query string into an AST.
pub fn parse(input: &str) -> Result<XPathQuery, ParseError> {
    let mut parser = Parser::new(input.trim());
    let query = parser.parse_query()?;
    parser.skip_ws();
    if !parser.remaining().is_empty() {
        return Err(parser.err(format!(
            "unexpected trailing input: '{}'",
            parser.remaining()
        )));
    }
    Ok(query)
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

/// A weighted node result from query evaluation.
#[derive(Debug, Clone)]
pub struct WeightedNode {
    pub node_id: String,
    pub node_type: String,
    pub name: String,
    pub weight: f64,
    /// Path from root to this node (list of node ids).
    pub path: Vec<String>,
}

/// Evaluate a parsed query against a tree.
///
/// `semantic_scorer` handles `[node≈"..."]` predicates. It receives
/// `(node_text_repr, query_string)` and returns a similarity score in `[0, 1]`.
pub fn evaluate(
    query: &XPathQuery,
    root: &TreeNode,
    semantic_scorer: &dyn Fn(&str, &str) -> f64,
) -> Vec<WeightedNode> {
    // Start with the root weighted 1.0.
    let mut working_set: Vec<(Vec<String>, &TreeNode, f64)> =
        vec![(vec![root.id.clone()], root, 1.0)];

    for step in &query.steps {
        let mut next_set: Vec<(Vec<String>, &TreeNode, f64)> = Vec::new();

        for (path, node, weight) in &working_set {
            // Axis expansion
            let candidates = match step.axis {
                Axis::Child => collect_children(node, path),
                Axis::Descendant => collect_descendants(node, path),
            };

            // Node-type filter
            let filtered: Vec<_> = candidates
                .into_iter()
                .filter(|(_, n, _)| match &step.node_select {
                    NodeSelect::Wildcard => true,
                    NodeSelect::Type(t) => n.node_type.as_str() == t,
                })
                .map(|(p, n, _)| (p, n, *weight))
                .collect();

            next_set.extend(filtered);
        }

        // Apply predicates
        for pred in &step.predicates {
            next_set = apply_predicate(pred, next_set, semantic_scorer);
        }

        working_set = next_set;
    }

    // Build results sorted by weight descending.
    let mut results: Vec<WeightedNode> = working_set
        .into_iter()
        .map(|(path, node, w)| WeightedNode {
            node_id: node.id.clone(),
            node_type: node.node_type.as_str().to_string(),
            name: node.name.clone(),
            weight: w,
            path,
        })
        .collect();
    results.sort_by(|a, b| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

fn collect_children<'a>(
    node: &'a TreeNode,
    parent_path: &[String],
) -> Vec<(Vec<String>, &'a TreeNode, f64)> {
    node.children
        .iter()
        .map(|c| {
            let mut p = parent_path.to_vec();
            p.push(c.id.clone());
            (p, c, 1.0)
        })
        .collect()
}

fn collect_descendants<'a>(
    node: &'a TreeNode,
    parent_path: &[String],
) -> Vec<(Vec<String>, &'a TreeNode, f64)> {
    let mut result = Vec::new();
    collect_descendants_rec(node, parent_path, &mut result);
    result
}

fn collect_descendants_rec<'a>(
    node: &'a TreeNode,
    parent_path: &[String],
    out: &mut Vec<(Vec<String>, &'a TreeNode, f64)>,
) {
    for child in &node.children {
        let mut p = parent_path.to_vec();
        p.push(child.id.clone());
        out.push((p.clone(), child, 1.0));
        collect_descendants_rec(child, &p, out);
    }
}

fn apply_predicate<'a>(
    pred: &Predicate,
    mut set: Vec<(Vec<String>, &'a TreeNode, f64)>,
    scorer: &dyn Fn(&str, &str) -> f64,
) -> Vec<(Vec<String>, &'a TreeNode, f64)> {
    match pred {
        Predicate::Position(i) => {
            let idx = if *i >= 0 {
                (*i - 1) as usize // 1-based to 0-based
            } else {
                let len = set.len() as i32;
                (len + *i) as usize
            };
            if idx < set.len() {
                vec![set.remove(idx)]
            } else {
                vec![]
            }
        }
        Predicate::Range(start, end) => {
            // 1-based inclusive range
            let s = (*start - 1).max(0) as usize;
            let e = (*end as usize).min(set.len());
            if s < e {
                set.into_iter().skip(s).take(e - s).collect()
            } else {
                vec![]
            }
        }
        Predicate::Semantic(query_text) => set
            .into_iter()
            .map(|(p, n, w)| {
                let score = scorer(&n.text_repr, query_text);
                (p, n, w * score)
            })
            .filter(|(_, _, w)| *w > 0.0)
            .collect(),
        Predicate::AttrEquals(key, value) => set
            .into_iter()
            .filter(|(_, n, _)| n.attributes.get(key.as_str()) == Some(value))
            .collect(),
        Predicate::Comparison(key, op, value) => {
            let target: f64 = value.parse().unwrap_or(f64::NAN);
            set.into_iter()
                .filter(|(_, n, _)| {
                    if let Some(attr_val) = n.attributes.get(key.as_str()) {
                        if let Ok(v) = attr_val.parse::<f64>() {
                            match op {
                                CompOp::Gt => v > target,
                                CompOp::Lt => v < target,
                                CompOp::Gte => v >= target,
                                CompOp::Lte => v <= target,
                                CompOp::Eq => (v - target).abs() < f64::EPSILON,
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                })
                .collect()
        }
        Predicate::Aggregate(op, subquery, _semantic_text) => {
            set.into_iter()
                .map(|(p, node, w)| {
                    // Evaluate sub-query on this node's subtree
                    let sub_results = evaluate(subquery, node, scorer);
                    let scores: Vec<f64> = sub_results.iter().map(|r| r.weight).collect();
                    let agg = if scores.is_empty() {
                        0.0
                    } else {
                        match op {
                            AggOp::Avg => scores.iter().sum::<f64>() / scores.len() as f64,
                            AggOp::Min => scores.iter().cloned().fold(f64::INFINITY, f64::min),
                            AggOp::Max => scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                            AggOp::GMean => {
                                let product: f64 = scores.iter().product();
                                product.powf(1.0 / scores.len() as f64)
                            }
                        }
                    };
                    (p, node, w * agg)
                })
                .filter(|(_, _, w)| *w > 0.0)
                .collect()
        }
        Predicate::And(a, b) => {
            let after_a = apply_predicate(a, set, scorer);
            apply_predicate(b, after_a, scorer)
        }
        Predicate::Or(a, b) => {
            let set_a = apply_predicate(a, set.clone(), scorer);
            let set_b = apply_predicate(b, set, scorer);
            // Merge, keeping best weight per node id
            let mut map: HashMap<String, (Vec<String>, &'a TreeNode, f64)> = HashMap::new();
            for item in set_a.into_iter().chain(set_b.into_iter()) {
                let entry = map.entry(item.1.id.clone()).or_insert_with(|| item.clone());
                if item.2 > entry.2 {
                    *entry = item;
                }
            }
            map.into_values().collect()
        }
        Predicate::Not(inner) => {
            let matched_ids: std::collections::HashSet<String> =
                apply_predicate(inner, set.clone(), scorer)
                    .iter()
                    .map(|(_, n, _)| n.id.clone())
                    .collect();
            set.into_iter()
                .filter(|(_, n, _)| !matched_ids.contains(&n.id))
                .collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Display impls (useful for debugging)
// ---------------------------------------------------------------------------

impl fmt::Display for XPathQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for step in &self.steps {
            match step.axis {
                Axis::Child => write!(f, "/")?,
                Axis::Descendant => write!(f, "//")?,
            }
            match &step.node_select {
                NodeSelect::Wildcard => write!(f, "*")?,
                NodeSelect::Type(t) => write!(f, "{}", t)?,
            }
            for pred in &step.predicates {
                write!(f, "[{}]", pred)?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Predicate::Position(i) => write!(f, "{}", i),
            Predicate::Range(a, b) => write!(f, "{}:{}", a, b),
            Predicate::Semantic(s) => write!(f, "node~\"{}\"", s),
            Predicate::AttrEquals(k, v) => write!(f, "{}=\"{}\"", k, v),
            Predicate::Comparison(k, op, v) => {
                let op_s = match op {
                    CompOp::Gt => ">",
                    CompOp::Lt => "<",
                    CompOp::Gte => ">=",
                    CompOp::Lte => "<=",
                    CompOp::Eq => "=",
                };
                write!(f, "{}{}{}", k, op_s, v)
            }
            Predicate::Aggregate(op, _q, s) => {
                let op_s = match op {
                    AggOp::Avg => "avg",
                    AggOp::Min => "min",
                    AggOp::Max => "max",
                    AggOp::GMean => "gmean",
                };
                write!(f, "{}(...~\"{}\")", op_s, s)
            }
            Predicate::And(a, b) => write!(f, "{} and {}", a, b),
            Predicate::Or(a, b) => write!(f, "{} or {}", a, b),
            Predicate::Not(inner) => write!(f, "not {}", inner),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: simple substring scorer for tests
    fn substring_scorer(text: &str, query: &str) -> f64 {
        let text_lower = text.to_lowercase();
        let query_lower = query.to_lowercase();
        if text_lower.contains(&query_lower) {
            1.0
        } else {
            0.0
        }
    }

    fn make_test_tree() -> TreeNode {
        // Project
        //   Session(auth) [agent=claude]
        //     Decision(use-jwt)
        //     ToolCall(login-api) [confidence=0.9]
        //     ToolCall(token-verify) [confidence=0.5]
        //   Session(refactor) [agent=gpt]
        //     Decision(extract-method)
        //     Memory(duckdb-schema) [confidence=0.85]

        let d1 = {
            let mut n = TreeNode::new("d1", TreeNodeType::Decision, "use-jwt");
            n.text_repr = "Decision to use JWT tokens for auth".into();
            n
        };
        let tc1 = {
            let mut n = TreeNode::new("tc1", TreeNodeType::ToolCall, "login-api")
                .with_attr("confidence", "0.9");
            n.text_repr = "Call login authentication API".into();
            n
        };
        let tc2 = {
            let mut n = TreeNode::new("tc2", TreeNodeType::ToolCall, "token-verify")
                .with_attr("confidence", "0.5");
            n.text_repr = "Verify auth token".into();
            n
        };
        let d2 = {
            let mut n = TreeNode::new("d2", TreeNodeType::Decision, "extract-method");
            n.text_repr = "Decision to extract method".into();
            n
        };
        let m1 = {
            let mut n = TreeNode::new("m1", TreeNodeType::Memory, "duckdb-schema")
                .with_attr("confidence", "0.85");
            n.text_repr = "DuckDB schema definition and usage".into();
            n
        };

        let mut s1 =
            TreeNode::new("s1", TreeNodeType::Session, "auth-session").with_attr("agent", "claude");
        s1.text_repr = "Authentication and login session".into();
        s1.children = vec![d1, tc1, tc2];

        let mut s2 = TreeNode::new("s2", TreeNodeType::Session, "refactor-session")
            .with_attr("agent", "gpt");
        s2.text_repr = "Code refactoring session".into();
        s2.children = vec![d2, m1];

        let mut root = TreeNode::new("root", TreeNodeType::Project, "test-project");
        root.text_repr = "Test project".into();
        root.children = vec![s1, s2];
        root
    }

    // -----------------------------------------------------------------------
    // Parser tests
    // -----------------------------------------------------------------------

    #[test]
    fn parse_descendant_with_semantic() {
        let q = parse(r#"//Session[node~"authentication"]"#).unwrap();
        assert_eq!(q.steps.len(), 1);
        assert_eq!(q.steps[0].axis, Axis::Descendant);
        assert_eq!(q.steps[0].node_select, NodeSelect::Type("Session".into()));
        assert!(
            matches!(&q.steps[0].predicates[0], Predicate::Semantic(s) if s == "authentication")
        );
    }

    #[test]
    fn parse_unicode_approx() {
        let q = parse("//Session[node\u{2248}\"authentication\"]").unwrap();
        assert!(
            matches!(&q.steps[0].predicates[0], Predicate::Semantic(s) if s == "authentication")
        );
    }

    #[test]
    fn parse_attr_filter_then_child() {
        let q = parse(r#"//Session[agent="claude"]/Decision"#).unwrap();
        assert_eq!(q.steps.len(), 2);
        assert_eq!(q.steps[0].axis, Axis::Descendant);
        assert!(
            matches!(&q.steps[0].predicates[0], Predicate::AttrEquals(k, v) if k == "agent" && v == "claude")
        );
        assert_eq!(q.steps[1].axis, Axis::Child);
        assert_eq!(q.steps[1].node_select, NodeSelect::Type("Decision".into()));
    }

    #[test]
    fn parse_negative_position() {
        let q = parse("//Project/Session[-1]/Decision").unwrap();
        assert_eq!(q.steps.len(), 3);
        assert!(matches!(&q.steps[1].predicates[0], Predicate::Position(-1)));
    }

    #[test]
    fn parse_aggregate() {
        let q = parse(r#"//Session[avg(/ToolCall[node~"auth"])]/Decision"#).unwrap();
        assert_eq!(q.steps.len(), 2);
        match &q.steps[0].predicates[0] {
            Predicate::Aggregate(AggOp::Avg, subq, sem) => {
                assert_eq!(subq.steps.len(), 1);
                assert_eq!(
                    subq.steps[0].node_select,
                    NodeSelect::Type("ToolCall".into())
                );
                assert_eq!(sem, "auth");
            }
            other => panic!("expected Aggregate, got: {:?}", other),
        }
    }

    #[test]
    fn parse_multiple_predicates() {
        let q = parse(r#"//Memory[node~"DuckDB"][confidence>0.8]"#).unwrap();
        assert_eq!(q.steps[0].predicates.len(), 2);
        assert!(matches!(&q.steps[0].predicates[0], Predicate::Semantic(s) if s == "DuckDB"));
        assert!(matches!(
            &q.steps[0].predicates[1],
            Predicate::Comparison(k, CompOp::Gt, v) if k == "confidence" && v == "0.8"
        ));
    }

    #[test]
    fn parse_wildcard() {
        let q = parse(r#"//*[node~"error handling"]"#).unwrap();
        assert_eq!(q.steps[0].node_select, NodeSelect::Wildcard);
    }

    #[test]
    fn parse_symbol() {
        let q = parse(r#"//Symbol[node~"parse_config"]"#).unwrap();
        assert_eq!(q.steps[0].node_select, NodeSelect::Type("Symbol".into()));
    }

    #[test]
    fn parse_and_predicate() {
        let q = parse(r#"//Session[node~"refactor" and agent="claude"]"#).unwrap();
        match &q.steps[0].predicates[0] {
            Predicate::And(a, b) => {
                assert!(matches!(a.as_ref(), Predicate::Semantic(s) if s == "refactor"));
                assert!(
                    matches!(b.as_ref(), Predicate::AttrEquals(k, v) if k == "agent" && v == "claude")
                );
            }
            other => panic!("expected And, got: {:?}", other),
        }
    }

    #[test]
    fn parse_range() {
        let q = parse("//Session[1:3]").unwrap();
        assert!(matches!(&q.steps[0].predicates[0], Predicate::Range(1, 3)));
    }

    #[test]
    fn parse_or_predicate() {
        let q = parse(r#"//Session[agent="claude" or agent="gpt"]"#).unwrap();
        assert!(matches!(&q.steps[0].predicates[0], Predicate::Or(_, _)));
    }

    #[test]
    fn parse_not_predicate() {
        let q = parse(r#"//Session[not agent="gpt"]"#).unwrap();
        assert!(matches!(&q.steps[0].predicates[0], Predicate::Not(_)));
    }

    // -----------------------------------------------------------------------
    // Parser error tests
    // -----------------------------------------------------------------------

    #[test]
    fn parse_error_empty() {
        assert!(parse("").is_err());
    }

    #[test]
    fn parse_error_no_slash() {
        assert!(parse("Session").is_err());
    }

    #[test]
    fn parse_error_unclosed_bracket() {
        assert!(parse("//Session[1").is_err());
    }

    #[test]
    fn parse_error_unclosed_string() {
        assert!(parse(r#"//Session[node~"unclosed]"#).is_err());
    }

    #[test]
    fn parse_error_trailing_input() {
        assert!(parse("//Session extra").is_err());
    }

    // -----------------------------------------------------------------------
    // Evaluator tests
    // -----------------------------------------------------------------------

    #[test]
    fn eval_descendant_semantic() {
        let tree = make_test_tree();
        let q = parse(r#"//Session[node~"authentication"]"#).unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "s1");
        assert_eq!(results[0].weight, 1.0);
    }

    #[test]
    fn eval_no_match_semantic() {
        let tree = make_test_tree();
        let q = parse(r#"//Session[node~"zzz_nonexistent"]"#).unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert!(results.is_empty());
    }

    #[test]
    fn eval_attr_filter_then_child() {
        let tree = make_test_tree();
        let q = parse(r#"//Session[agent="claude"]/Decision"#).unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "d1");
    }

    #[test]
    fn eval_positional_last() {
        let tree = make_test_tree();
        let q = parse("//Session[-1]").unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "s2");
    }

    #[test]
    fn eval_positional_first() {
        let tree = make_test_tree();
        let q = parse("//Session[1]").unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "s1");
    }

    #[test]
    fn eval_range() {
        let tree = make_test_tree();
        let q = parse("//Session[1:2]").unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn eval_child_axis() {
        // Test child axis from root (Project) to Session children
        let tree = make_test_tree();
        let q = parse("/Session").unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn eval_comparison_gt() {
        let tree = make_test_tree();
        let q = parse("//ToolCall[confidence>0.8]").unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "tc1");
    }

    #[test]
    fn eval_comparison_lte() {
        let tree = make_test_tree();
        let q = parse("//ToolCall[confidence<=0.5]").unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "tc2");
    }

    #[test]
    fn eval_multiple_predicates() {
        let tree = make_test_tree();
        let q = parse(r#"//Memory[node~"DuckDB"][confidence>0.8]"#).unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "m1");
    }

    #[test]
    fn eval_wildcard() {
        let tree = make_test_tree();
        let q = parse(r#"//*[node~"auth"]"#).unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        // Should find: s1 (authentication), d1 (auth), tc1 (authentication), tc2 (auth)
        assert!(results.len() >= 2);
        let ids: Vec<&str> = results.iter().map(|r| r.node_id.as_str()).collect();
        assert!(ids.contains(&"s1"));
        assert!(ids.contains(&"tc1"));
    }

    #[test]
    fn eval_and_predicate() {
        let tree = make_test_tree();
        let q = parse(r#"//Session[node~"refactor" and agent="gpt"]"#).unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "s2");
    }

    #[test]
    fn eval_and_no_match() {
        let tree = make_test_tree();
        let q = parse(r#"//Session[node~"refactor" and agent="claude"]"#).unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert!(results.is_empty());
    }

    #[test]
    fn eval_or_predicate() {
        let tree = make_test_tree();
        let q = parse(r#"//Session[agent="claude" or agent="gpt"]"#).unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn eval_not_predicate() {
        let tree = make_test_tree();
        let q = parse(r#"//Session[not agent="gpt"]"#).unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "s1");
    }

    #[test]
    fn eval_aggregate_avg() {
        let tree = make_test_tree();
        let q = parse(r#"//Session[avg(/ToolCall[node~"auth"])]"#).unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        // s1 has ToolCall children matching "auth" (tc1 and tc2 both contain "auth").
        // Both score 1.0 via substring match => avg = 1.0 => s1 weight = 1.0.
        // s2 has no ToolCall children => avg = 0.0 => filtered out.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "s1");
    }

    #[test]
    fn eval_path_tracking() {
        let tree = make_test_tree();
        let q = parse(r#"//Session[agent="claude"]/Decision"#).unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert_eq!(results.len(), 1);
        // Path should be root -> s1 -> d1
        assert_eq!(results[0].path, vec!["root", "s1", "d1"]);
    }

    #[test]
    fn eval_empty_result() {
        let tree = make_test_tree();
        let q = parse("//Symbol").unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // @attr syntax tests (standard XPath attribute prefix)
    // -----------------------------------------------------------------------

    #[test]
    fn parse_at_attr_filter() {
        // @attr=value should parse identically to attr=value
        let with_at = parse(r#"//Session[@agent="claude"]"#).unwrap();
        let without_at = parse(r#"//Session[agent="claude"]"#).unwrap();
        assert_eq!(format!("{with_at:?}"), format!("{without_at:?}"));
    }

    #[test]
    fn parse_at_attr_comparison() {
        let q = parse(r#"//ToolCall[@confidence>0.8]"#).unwrap();
        let step = &q.steps[0];
        assert_eq!(step.predicates.len(), 1);
        match &step.predicates[0] {
            Predicate::Comparison(attr, CompOp::Gt, val) => {
                assert_eq!(attr, "confidence");
                assert_eq!(val, "0.8");
            }
            other => panic!("expected Comparison, got {other:?}"),
        }
    }

    #[test]
    fn eval_at_attr_filter_then_child() {
        let tree = make_test_tree();
        let q = parse(r#"//Session[@agent="claude"]/Decision"#).unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "d1");
    }

    #[test]
    fn parse_at_attr_in_and_predicate() {
        let q = parse(r#"//Session[@agent="claude" and node~"auth"]"#).unwrap();
        let step = &q.steps[0];
        assert_eq!(step.predicates.len(), 1);
        match &step.predicates[0] {
            Predicate::And(_, _) => {} // expected
            other => panic!("expected And, got {other:?}"),
        }
    }

    #[test]
    fn parse_at_attr_in_or_predicate() {
        let q = parse(r#"//Session[@agent="claude" or @agent="gpt"]"#).unwrap();
        let step = &q.steps[0];
        match &step.predicates[0] {
            Predicate::Or(_, _) => {}
            other => panic!("expected Or, got {other:?}"),
        }
    }

    #[test]
    fn eval_at_attr_comparison_lte() {
        let tree = make_test_tree();
        let q = parse("//ToolCall[@confidence<=0.5]").unwrap();
        let results = evaluate(&q, &tree, &substring_scorer);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "tc2");
    }
}
