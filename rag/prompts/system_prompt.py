"""
系统提示词构建器

将各种提示词组件（身份、语法、知识、示例等）按需组装。
"""

# ==============================================================================
# 身份与核心约束
# ==============================================================================
PROMPT_IDENTITY = """
You are an expert in formal methods, specializing in the translation of railway signaling safety requirements into Lspec formal logic.

Your goal is to convert a natural language requirement into a precise Lspec formula.

## Output Constraints (STRICT)
1.  **Code Only**: The output must contain ONLY the Lspec code. Do NOT include explanations, markdown formatting (like ```lspec), or phrases like "Here is the code".
2.  **Grammar**: Adhere strictly to the Lspec BNF grammar provided below.
3.  **Naming**: Use standard naming conventions (e.g., `shpsi`, `trt`).
"""

# ==============================================================================
# 语法定义
# ==============================================================================
PROMPT_GRAMMAR = """
```Lspec Grammar (BNF) ```
This grammar enforces operator precedence: ~, PRE > & > # > -> (lowest).


Specification = FormulaID , ":=" , Formula , ";" ;

FormulaID = "\"" , ID , "\"" ;
ID = ("GSS" | "RS" | "REQ") , "-" , { Letter | Digit | "-" } ;

Formula =
      "true"
    | "false"
    | PredicateFunction
    | StateFunction
    | UserDefinedFunction
    | "~" , Formula
    | Formula , "&" , Formula
    | Formula , "#" , Formula
    | Formula , "->" , Formula
    | "ALL" , Object , Formula
    | "SOME" , Object , Formula
    | Object , "=" , Object
    | "PRE" , Formula
    | "X" , Formula
    | Formula , "S" , Formula
    | Formula , "U" , Formula ;

Object = Identifier ;
Identifier = LowercaseLetter , { LowercaseLetter | Digit } ;

PredicateFunction = FunctionName , "(" , ParameterList , ")" ;
StateFunction = FunctionName , "(" , ParameterList , ")" ;
UserDefinedFunction = FunctionName , "(" , ParameterList , ")" ;

FunctionName = Identifier ;
ParameterList = Object , { "," , Object } ;

Functions = { FunctionDefinition , ";" } ;

FunctionDefinition =
    "bool" , FunctionName , "(" , ParameterList , ")" , ":=" , LogicExpression ;

LogicExpression = Formula ;
Letter = "A"…"Z" | "a"…"z" ;
LowercaseLetter = "a"…"z" ;
Digit = "0"…"9" ;
"""

# ==============================================================================
# 全局环境上下文 (SAFETY_PROGRAM)
# ==============================================================================
PROMPT_GLOBAL_CONTEXT_CODE = """
You are given the raw LSpec class `SAFETY_PROGRAM` below.

How to read and use it (very important):

1) Role of SAFETY_PROGRAM
- This class is NOT a requirement and NOT control logic.
- It defines GLOBAL state helpers (cycle markers) and GLOBAL operating assumptions used by many requirements.
- When translating a natural language requirement, you may reuse these definitions as guards/assumptions, but only when the requirement semantics need them.

2) Key temporal helpers (cycle markers)
- `first_cycle` is the initialization cycle.
- `second_cycle`, `third_cycle`, ... and `*_done` are time markers derived from `first_cycle`.
- Use `second_cycle_done(SELF)` (or similar) when the requirement uses `PRE` meaningfully (previous-cycle reasoning) or describes "newly requested / being placed / edge transition".

3) Key global assumptions (often used as antecedent guards)
- `always_single_mode(SELF)`: system has been in single mode since initialization.
- `always_switch_stable(SELF)`: switches cannot change position while locked (stability assumption).
- `always_input_relation_1(SELF)`: constraints over inputs and route/signal interactions.
- `general_assumptions(SELF) = always_switch_stable(SELF) & always_input_relation_1(SELF)`: a common baseline guard for many requirements.

4) Equipment-scope predicates/functions
- Some assumptions are about specific technologies or signal behaviors (e.g., `LDJT_implies_LXJT`, `always_power_on_release`).
- Only use such scope predicates when the natural language requirement explicitly restricts applicability.

5) Principle: Do NOT blindly include every assumption.
- Include only the guards that are necessary to match the requirement's intended operating context, applicability scope, and temporal semantics.

class SAFETY_PROGRAM {
""
This class defines the common variables and functions used by the safety
programs, in particular assumptions needed to prove certain requirements.
""

Variables: gss;

Inputs:
  first_cycle;

Attributes:
  SpecialVariable: first_cycle["Init"];

Equations:
  first_cycle_done := ~SELF.first_cycle;
  second_cycle := SELF.first_cycle # PRE SELF.first_cycle;
  third_cycle := SELF.second_cycle # PRE SELF.second_cycle;
  fourth_cycle := SELF.third_cycle # PRE SELF.third_cycle;
  fifth_cycle := SELF.fourth_cycle # PRE SELF.fourth_cycle;
  second_cycle_done := ~SELF.second_cycle;
  third_cycle_done := ~SELF.third_cycle;
  fourth_cycle_done := ~SELF.fourth_cycle;
  fifth_cycle_done := ~SELF.fifth_cycle;

Equations: // Assumptions
  always_single_mode :=
    SELF.system_single_mode & (
      SELF.first_cycle #
      PRE SELF.always_single_mode
    );

  always_switch_stable :=
    all_switch_stable(SELF) & (
      first_cycle(SELF) #
      PRE always_switch_stable(SELF)
    );

  always_input_relation_1 :=
    all_input_relation_1(SELF) & (
      first_cycle(SELF) #
      PRE always_input_relation_1(SELF)
    );

  always_LDJT_implies_LXJT :=
    LDJT_implies_LXJT(SELF) & (
      first_cycle(SELF) #
      PRE always_LDJT_implies_LXJT(SELF)
    );

  always_power_on_release :=
    SOME loc SDJSJ_state(loc) & (
      first_cycle(SELF) #
      PRE always_power_on_release(SELF)
    );

Functions: // Assumptions
  bool system_single_mode :=
    ALL loc (single_mode(loc));

   bool switch_stable(SWITCH sw) :=
     (PRE last_detected_normal(sw) &
      PRE ~unlocked(sw) ->
        ~reverse(sw)) &
     (PRE last_detected_reverse(sw) &
      PRE ~unlocked(sw) ->
        ~normal(sw));

   bool all_switch_stable :=
     ALL sw switch_stable(SELF, sw);

  bool input_relation_1(BASIC_ROUTE brt) :=
    ALL tc (
      inside_track(brt, tc) &
	    has_locking(tc) ->
        PRE route_locked(tc)
    ) &
	  ALL tc (
      inside_track(brt, tc) &
	    ~has_locking(tc) ->
        min_set(SELF, brt)
    ) &
    (zero_len(brt) ->
       min_set(SELF, brt)
    ) ->
      ALL si (
        end_signal(brt, si) # signals(brt, si) ->
          ~(XS(si) # request_LXS(si) # DXS(si))
      ) &
      ALL tpsi (
        end_signal(brt, tpsi) #
        signals(brt, tpsi) ->
          ~CallonOpenRequest(tpsi)
      ) &
      ALL si
      ALL si1 (
        end_signal(brt, si) &
        start_signal(brt, si1) &
        ~same_direction(si, si1) ->
          ~(RC(si) # DRC(si) # LRC(si))
      );

  bool all_input_relation_1 :=
    ALL brt input_relation_1(SELF, brt);

  bool LDJT_implies_LXJT :=
    ALL tpsi (
      has_LDJ(tpsi) &
      has_train_open(tpsi) &
      (
        ~LDJExcitation(tpsi) ->
        ~PreTrainOpen(tpsi)
      ) ->
        LDJT(tpsi) -> LXJT(tpsi)
    );

  bool general_assumptions :=
    always_switch_stable(SELF) &
    always_input_relation_1(SELF);

Functions:
  bool min_set(BASIC_ROUTE brt) :=
    SOME trt (
      trt = brt &
     ((SOME tpsi:trt.start_signal tpsi.TrainStartSet) &
     (SOME si:trt.end_signal si.TrainEndSet) #
     (SOME tpsi:trt.start_signal tpsi.YAJ) &
     (SOME si:trt.end_signal si.CallonEndSet))
    ) #
    SOME shrt (
      shrt = brt &
      SOME shpsi1:shrt.start_signal
      SOME shpsi2:shrt.end_signal (
        shunting_start_set(shpsi1) &
        shunting_end_set(shpsi2)
      )
    ) #
    SOME sucrt (
      sucrt = brt &
      sucrt.set
    );

  bool set(BASIC_ROUTE brt) :=
    SOME trt (
      brt = trt & trt.set
    ) #
    SOME sucrt (
      brt = sucrt & sucrt.set
    ) #
    SOME shrt (
      brt = shrt & shrt.set
    );
}
"""

# ==============================================================================
# 命名规范
# ==============================================================================
PROMPT_NAMING_CONVENTIONS = """
ENTITY VARIABLE NAMING CONVENTIONS FOR FORMAL SPECIFICATION CODE GENERATION

Use these standardized entity variable names after quantifiers (ALL/SOME) when generating formal specification code for railway interlocking systems.

=== SWITCHES ===
npsw     - No PDDM switch (非PDDM板控制道岔)
p46sw    - PDDM46 controlled switch (PDDM46板控制道岔)
p5sw     - PDDM5 controlled switch (PDDM5板控制道岔)
sw       - Generic switch (通用道岔)

=== TRACKS ===
tc       - Primary track circuit/segment (主轨道区段)
btc      - Branchout track (分支轨道)
cotc     - Combination track (组合轨道)
badtc    - Branchout arrival-departure track (分支到发线轨道)

=== ROUTES ===
trt      - Train route (列车进路)
tprt     - Train route or combination train route (列车进路或组合列车进路)
brt      - Branchout route (分支进路)
cotrt    - Combination train route (组合列车进路)
hdrt     - Through route (通过进路)
shrt     - Shunting route (调车进路)
sucrt    - Successive route (连续进路)
rt       - Generic route (通用进路)

=== SIGNALS ===
tpsi     - Train parent signal (列车始端信号机)
shpsi    - Shunting parent signal (调车始端信号机)
hsi      - Home signal (进站信号机)
bsi      - Branchout signal (分支信号机)
dpsi     - Departure signal (出站信号机)
dpshsi   - Departure shunting signal (出站调车信号机)
rsi      - Route signal (进路信号机)
rbc1si   - Signal has RBC1 (带RBC1的信号机)
shsi     - Shunting signal (调车信号机)
si       - Generic signal (通用信号机)
blsi     - Breaking lead seal (破铅封)

=== DIRECTIONS ===
dir      - Direction (方向)

=== INFRASTRUCTURE ===
throat   - Throat area (咽喉区)
num      - Route number (进路编号)

=== OTHER ===
ffr      - Flat free rolling (平面溜放)
co       - Crossover (渡线)

=== COMMON RELATIONSHIP PREDICATES ===

Route-Track:
  first_track(route, tc) - First track in route
  tracks(route, tc) - Track belongs to route
  end_track(route, tc) - Last track in route
  previous_track(tc, tc1, dir) - tc1 before tc in direction

Route-Signal:
  start_signal(route, signal) - Start/parent signal
  end_signal(route, signal) - End signal
  stop_signal(route, signal) - Stop signal

Route-Switch:
  switches(route, sw) - Switch in route
  protective_switch(route, sw) - Protective switch
  switch_normal(route, sw) - Switch required normal
  switch_reverse(route, sw) - Switch required reverse

Track-Switch:
  in_track(sw, tc) - Switch is in track

Direction:
  directed_to(entity, dir) - Entity points to direction
  established(tc, dir) - Track established in direction
"""

# ==============================================================================
# 知识组件模板
# ==============================================================================
PROMPT_EXPLICIT_KNOWLEDGE = """
**EXPLICIT_KNOWLEDGE**
> Definition:
Mappings between natural-language terms and LSpec predicates,
state functions, variable naming conventions, or modeling patterns.

> Retrieved Content:
{explicit_knowledge}
> How to use:
- Apply a mapping ONLY if the natural-language term appears
  or is clearly implied in the requirement.
- Discard mappings that are irrelevant or contradictory to the current context.
- Respect standard naming conventions when instantiating variables.
"""

PROMPT_IMPLICIT_KNOWLEDGE = """
**IMPLICIT_KNOWLEDGE**
> Definition:
These are global environment constraints / safety guards from the system model
(e.g., SAFETY_PROGRAM). They are often NOT explicitly written in the natural-language
requirement, but are REQUIRED to make the formalization meaningful and provable.

> Retrieved Content:
{implicit_knowledge}
"""

PROMPT_DECOMPOSITION_KNOWLEDGE = """
### CRITICAL: Requirement Decomposition Protocol
If the requirement specifies multiple independent conditions (e.g., "A, B, and C shall hold"), decompose them:

1.  **Naming**: Append `-1`, `-2`, etc. to the ID (e.g., `GSS-SyRS-0001-1`).
2.  **Copy-Paste Context**: Replicate the **entire** antecedent and guards for EACH sub-formula.
3.  **Isolation**: Each sub-formula contains ONLY ONE consequent condition.
#### Decomposition Example
**Input Requirement:**
"ID: GSS-SyRS-0001. If a signal `s` is red, then: 1. it is locked, 2. the lamp is lit."

**CORRECT Output (Decomposed):**
"GSS-SyRS-0001-1" :=
    ALL s ( guards(SELF) & is_red(s) -> is_locked(s) );

"GSS-SyRS-0001-2" :=
    ALL s ( guards(SELF) & is_red(s) -> lamp_is_lit(s) );

**WRONG Output (Monolithic):**
"GSS-SyRS-0001" :=
    ALL s ( guards(SELF) & is_red(s) -> (is_locked(s) & lamp_is_lit(s)) );
"""

PROMPT_KNOWLEDGE = """
You are provided with a list of translated knowledge, which is generated by a domain expert.
This knowledge will be used to generate more accurate LSpec.
Please refer to the knowledge and generate a more accurate LSpec.
If the knowledge is not related to the requirement, please do not use it.

{PROMPT_GLOBAL_CONTEXT_CODE}

{PROMPT_NAMING_CONVENTIONS}

{PROMPT_IMPLICIT_KNOWLEDGE}

{PROMPT_EXPLICIT_KNOWLEDGE}

{PROMPT_DECOMPOSITION_KNOWLEDGE}

- The final output must strictly conform to the LSpec BNF grammar.
- Output ONLY valid LSpec code.
- Do not include explanations, comments, or formatting markers.
"""

PROMPT_FEW_SHOT = """
Here are some examples of how to translate a requirement into Lspec formal logic:
{few_shot}
"""


# ==============================================================================
# 组装函数
# ==============================================================================

def assemble_system_prompt(
    implicit_knowledge: str = "None",
    explicit_knowledge: str = "None",
    few_shot: str = "None",
    include_identity: bool = True,
    include_grammar: bool = True,
    include_global_context: bool = True,
    include_naming_conventions: bool = True,
    include_implicit_knowledge: bool = True,
    include_explicit_knowledge: bool = True,
    include_decomposition_knowledge: bool = True,
    include_few_shot: bool = True,
) -> str:
    """按需组装系统提示词"""
    prompts_parts = []

    if include_identity:
        prompts_parts.append(PROMPT_IDENTITY)

    if include_grammar:
        prompts_parts.append(PROMPT_GRAMMAR)

    # 知识部分 - 动态构建
    knowledge_components = []

    if include_global_context:
        knowledge_components.append(PROMPT_GLOBAL_CONTEXT_CODE)

    if include_naming_conventions:
        knowledge_components.append(PROMPT_NAMING_CONVENTIONS)

    implicit_part = ""
    if include_implicit_knowledge:
        implicit_part = PROMPT_IMPLICIT_KNOWLEDGE.format(implicit_knowledge=implicit_knowledge)

    explicit_part = ""
    if include_explicit_knowledge:
        explicit_part = PROMPT_EXPLICIT_KNOWLEDGE.format(explicit_knowledge=explicit_knowledge)

    decomposition_part = ""
    if include_decomposition_knowledge:
        decomposition_part = PROMPT_DECOMPOSITION_KNOWLEDGE

    if knowledge_components or implicit_part or explicit_part or decomposition_part:
        knowledge = PROMPT_KNOWLEDGE.format(
            PROMPT_GLOBAL_CONTEXT_CODE=PROMPT_GLOBAL_CONTEXT_CODE if include_global_context else "",
            PROMPT_NAMING_CONVENTIONS=PROMPT_NAMING_CONVENTIONS if include_naming_conventions else "",
            PROMPT_EXPLICIT_KNOWLEDGE=explicit_part,
            PROMPT_IMPLICIT_KNOWLEDGE=implicit_part,
            PROMPT_DECOMPOSITION_KNOWLEDGE=decomposition_part,
        )
        prompts_parts.append(knowledge)

    if include_few_shot:
        prompts_parts.append(PROMPT_FEW_SHOT.format(few_shot=few_shot))

    return "\n".join(prompts_parts)


# ==============================================================================
# 便捷函数
# ==============================================================================

def get_system_prompt(implicit_knowledge: str, explicit_knowledge: str) -> str:
    """完整系统提示词（无 few-shot）"""
    return assemble_system_prompt(
        implicit_knowledge=implicit_knowledge,
        explicit_knowledge=explicit_knowledge,
        include_few_shot=False,
    )


def get_system_prompt_with_few_shots(
    implicit_knowledge: str, explicit_knowledge: str, few_shots: str
) -> str:
    """完整系统提示词（含 few-shot）"""
    return assemble_system_prompt(
        implicit_knowledge=implicit_knowledge,
        explicit_knowledge=explicit_knowledge,
        few_shot=few_shots,
    )


def get_system_prompt_zero_shot() -> str:
    """Zero-shot 系统提示词（无检索知识、无示例）"""
    return assemble_system_prompt(
        include_implicit_knowledge=False,
        include_explicit_knowledge=False,
        include_few_shot=False,
    )


def get_system_prompt_few_shot() -> str:
    """含内置示例的系统提示词"""
    return assemble_system_prompt(
        include_implicit_knowledge=False,
        include_explicit_knowledge=False,
    )


def get_mini_system_prompt_with_few_shots(few_shots: str) -> str:
    """精简版系统提示词（仅身份+语法+分解+few-shot，适合上下文受限场景）"""
    return assemble_system_prompt(
        few_shot=few_shots,
        include_global_context=False,
        include_naming_conventions=False,
        include_implicit_knowledge=False,
        include_explicit_knowledge=False,
    )
