-- Example Lean 4 statements

-- Simple theorem about natural numbers
theorem add_zero (n : Nat) : n + 0 = n := by
  rfl

-- Function definition with pattern matching
def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Type class instance
instance : Add Nat where
  add := Nat.add

-- Structure definition
structure Point where
  x : Float
  y : Float
  deriving Repr

-- Inductive type definition
inductive Tree (α : Type)
  | leaf : α → Tree α
  | node : Tree α → Tree α → Tree α

-- Theorem with tactics
theorem not_not_intro {p : Prop} (h : p) : ¬¬p := by
  intro hn
  contradiction
