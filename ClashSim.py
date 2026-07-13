import math, random
import pygame
from dataclasses import dataclass, field

# ====================
# Config
# ====================
W_TILES = 18
H_TILES = 32

BRIDGE_W_TILES = 3

RIVER_Y0 = H_TILES // 2 - 1 # 15
RIVER_Y1 = H_TILES // 2 # 16
RIVER_ROWS = {RIVER_Y0, RIVER_Y1}

# Two lanes, left 0...8, right 9...17
LANE_SPLIT_X = 9
LEFT_LANE_CENTER_X = 4
RIGHT_LANE_CENTER_X = 13

# Tick rate
TPS = 20
DT = 1.0 / TPS

# Rendering
TILE_PX = 26
MARGIN_PX = 14
UI_HEIGHT_PX = 80

SCREEN_W = MARGIN_PX * 2 + TILE_PX * W_TILES
SCREEN_H = MARGIN_PX * 2 + TILE_PX * H_TILES + UI_HEIGHT_PX

# Teams
P0 = 0 # top player
P1 = 1 # bottom player

# Reward shaping (used by step()'s RL API)
SHAPE_SCALE = 1.0 / 500.0 # tower-HP damage -> reward units
WIN_REWARD = 10.0         # terminal bonus/penalty for winning/losing
TRADE_SCALE = 1.0 / 10.0  # net enemy elixir-value destroyed -> reward units
SPEND_SCALE = 0.15        # penalty per elixir spent: a play must earn its cost (anti-dump)
OVERFLOW_SCALE = 0.5      # penalty per elixir wasted at the 10-cap (anti-hoard)

def clamp(v: float, lo: float, hi: float):
    return lo if v < lo else hi if v > hi else v

def tile_to_px(tx: float, ty: float) -> tuple[int, int]:
    """Convert tile coordinate (top-left corner of tile) to pixel-space representing center of tile"""
    x = MARGIN_PX + int((tx + 0.5) * TILE_PX)
    y = MARGIN_PX + int((ty + 0.5) * TILE_PX)
    return x, y


# ====================
# Game objects
# ====================

@dataclass
class Tower:
    team: int
    kind: str # "crown" or "king"
    rect: pygame.Rect
    hp: float
    max_hp: float

    @property
    def center(self) -> tuple[float, float]:
        return self.rect.x + self.rect.w / 2, self.rect.y + self.rect.h / 2

@dataclass
class UnitDef:
    name: str
    cost: int
    max_hp: float
    speed_tiles_per_sec: float
    damage: float
    attack_range_tiles: float
    attacks_per_sec: float
    targets_buildings: bool = False # e.g. Giant: ignores troops, walks to towers

@dataclass
class Unit:
    uid: int
    team: int
    udef: UnitDef
    x: float
    y: float
    hp: float
    lane: int # 0 left, 1 right
    atk_cd: float
    target_unit: int | None = None # uid
    target_tower: tuple[int, str] | None = None # team, kind

@dataclass
class CardDef:
    """A playable card. Either a troop (spawns `count` units of `unit_def`) or a
    spell (`is_spell=True`: instant area damage at the target, no unit spawned)."""
    name: str
    cost: int
    unit_def: UnitDef | None = None
    count: int = 1                 # troops spawned per play (e.g. Goblins = 3)
    is_spell: bool = False
    spell_radius: float = 0.0      # tiles
    spell_damage: float = 0.0      # to units inside the radius
    spell_tower_damage: float = 0.0 # reduced damage to towers inside the radius

    @property
    def targets_buildings(self) -> bool:
        return self.unit_def is not None and self.unit_def.targets_buildings

@dataclass
class PlayerState:
    elixir: float = 7.0
    elixir_regen: float = 1.0 / 2.8 # per sec
    hand: list[str] = field(default_factory=list)
    deck: list[str] = field(default_factory=list)
    deck_index: int = 0

@dataclass
class GameState:
    tick: int = 0
    time_left: float = 180.0
    units: dict[int, Unit] = field(default_factory=dict) # uid, unit
    next_uid: int = 1
    players: list[PlayerState] = field(default_factory=lambda: [PlayerState(), PlayerState()])
    towers: list[Tower] = field(default_factory=list)
    winner: int | None = None # 0 or 1


# ====================
# Clash Simulator
# ====================

class MiniClash:
    def __init__(self):
        self.unit_defs = self._make_unit_defs()
        self.card_defs = self._make_card_defs()
        self.state = self._new_game()

    @staticmethod
    def _make_unit_defs() -> dict[str, UnitDef]:
        return {
            "knight": UnitDef(
                name="knight",
                cost=3,
                max_hp=600,
                speed_tiles_per_sec=2.2,
                damage=120,
                attack_range_tiles=1.2,
                attacks_per_sec=1.0),
            "archer": UnitDef(
                name="archer",
                cost=3,
                max_hp=300,
                speed_tiles_per_sec=2.0,
                damage=80,
                attack_range_tiles=4.5,
                attacks_per_sec=1.0),
            "giant": UnitDef(
                name="giant",
                cost=5,
                max_hp=2200,
                speed_tiles_per_sec=1.3,   # slow tank
                damage=140,
                attack_range_tiles=1.2,
                attacks_per_sec=0.8,
                targets_buildings=True),   # ignores troops, marches on towers
            "goblin": UnitDef(
                name="goblin",
                cost=1,                    # per-unit trade value (~ 2-cost card / 3)
                max_hp=120,
                speed_tiles_per_sec=3.0,   # fast, fragile swarm
                damage=70,
                attack_range_tiles=1.0,
                attacks_per_sec=1.4),
        }

    def _make_card_defs(self) -> dict[str, CardDef]:
        return {
            "Knight": CardDef(name="Knight", cost=3, unit_def=self.unit_defs["knight"]),
            "Archer": CardDef(name="Archer", cost=3, unit_def=self.unit_defs["archer"]),
            "Giant": CardDef(name="Giant", cost=5, unit_def=self.unit_defs["giant"]),
            "Goblins": CardDef(name="Goblins", cost=2, unit_def=self.unit_defs["goblin"], count=3),
            "Arrows": CardDef(name="Arrows", cost=3, is_spell=True,
                              spell_radius=4.0, spell_damage=240, spell_tower_damage=72),
        }

    @staticmethod
    def _new_game() -> GameState:
        st = GameState()

        def add_tower(team: int, kind: str, x: int, y: int, w: int, h: int, hp: float):
            st.towers.append(Tower(team=team, kind=kind, rect=pygame.Rect(x, y, w, h), hp=hp, max_hp=hp))

        # P0 (top player)
        add_tower(0, "crownL", x=2, y=4, w=3, h=3, hp=2000)
        add_tower(0, "crownR", x=13, y=4, w=3, h=3, hp=2000)
        add_tower(0, "king", x=7, y=1, w=4, h=4, hp=3500)

        # P1 (bottom player)
        add_tower(1, "crownL", x=2, y=H_TILES-7, w=3, h=3, hp=2000)
        add_tower(1, "crownR", x=13, y=H_TILES-7, w=3, h=3, hp=2000)
        add_tower(1, "king", x=7, y=H_TILES-4, w=4, h=4, hp=3500)

        base_deck = ["Knight", "Archer", "Giant", "Goblins", "Arrows", "Knight", "Archer", "Goblins"]
        for p in (P0, P1):
            st.players[p].deck = base_deck[:]
            random.shuffle(st.players[p].deck)
            st.players[p].hand = [st.players[p].deck[i] for i in range(4)]
            st.players[p].deck_index = 4

        return st

    # --- RL API ---
    def reset(self) -> GameState:
        self.state = self._new_game()
        return self.state

    def step(self, actions: dict[int, tuple[int, int, int] | None]) -> tuple[GameState, dict[int, float], bool]:
        """
        :param actions: {team : (hand slot, spawn_x, spawn_y) or None}
        :return: new GameState, {team: reward}, done
        """
        st = self.state
        # Check winner
        if st.winner is not None:
            return st, {P0: 0.0, P1: 0.0}, True

        # Snapshot tower HP and living units so we can reward the swing this tick caused
        # (tower damage) plus the elixir value of units killed on each side (fair trades).
        pre_hp = {P0: self._team_tower_hp(P0), P1: self._team_tower_hp(P1)}
        pre_units = {uid: (u.team, u.udef.cost) for uid, u in st.units.items()}
        pre_elixir = {team: st.players[team].elixir for team in (P0, P1)}

        # 1) apply actions
        for team, act in actions.items():
            if act is None:
                continue
            hand_slot, sx, sy = act
            self._try_play_card(team, hand_slot, sx, sy)

        spent = {team: pre_elixir[team] - st.players[team].elixir for team in (P0, P1)}

        # 2) regen elixir (track elixir wasted against the 10-cap)
        overflow = {P0: 0.0, P1: 0.0}
        for team in (P0, P1):
            ps = st.players[team]
            raw = ps.elixir + ps.elixir_regen * DT
            overflow[team] = max(0.0, raw - 10.0)
            ps.elixir = clamp(raw, 0.0, 10.0)

        # 3) update units
        self._update_units()

        # 4) time ticks and bookkeeping
        st.tick += 1
        st.time_left = max(0, st.time_left - DT)

        # 5)
        self._check_winner()

        # Elixir value of units that died this tick, per team.
        killed_value = {P0: 0.0, P1: 0.0}
        for uid, (team, cost) in pre_units.items():
            if uid not in st.units:
                killed_value[team] += cost

        done = (st.winner is not None) or st.time_left <= 0
        rewards = self._compute_rewards(pre_hp, killed_value, done)
        # Efficiency penalties: discourage dumping elixir on low-value plays and
        # discourage wasting regen at the cap. Together these make "hold for a strong
        # play" a learnable option instead of spamming whatever is affordable.
        for team in (P0, P1):
            rewards[team] -= spent[team] * SPEND_SCALE + overflow[team] * OVERFLOW_SCALE
        return st, rewards, done

    def _team_tower_hp(self, team: int) -> float:
        return sum(max(0.0, t.hp) for t in self.state.towers if t.team == team)

    def _compute_rewards(self, pre_hp: dict[int, float],
                         killed_value: dict[int, float], done: bool) -> dict[int, float]:
        """Dense shaped reward, plus a terminal win/loss bonus. Each tick a team is
        rewarded for (a) net tower damage dealt and (b) net enemy elixir-value destroyed
        in fights. The trade term is what lets an agent learn that defending efficiently
        -- killing more value than it spends -- pays off and sets up a counterpush.
        """
        st = self.state
        rewards: dict[int, float] = {}
        for team in (P0, P1):
            enemy = self._enemy_team(team)
            dmg_to_enemy = pre_hp[enemy] - self._team_tower_hp(enemy)
            dmg_to_self = pre_hp[team] - self._team_tower_hp(team)
            trade = killed_value[enemy] - killed_value[team]
            rewards[team] = (dmg_to_enemy - dmg_to_self) * SHAPE_SCALE + trade * TRADE_SCALE

        if done:
            for team in (P0, P1):
                if st.winner is None:
                    continue  # draw / timeout: shaped reward already reflects tower diff
                rewards[team] += WIN_REWARD if st.winner == team else -WIN_REWARD
        return rewards

    def _try_play_card(self, team: int, hand_slot: int, sx: int, sy: int) -> bool:
        st = self.state
        ps = st.players[team]

        if hand_slot < 0 or hand_slot >= len(ps.hand):
            return False

        if not (0 <= sx < W_TILES and 0 <= sy < H_TILES):
            return False

        card_name = ps.hand[hand_slot]
        cdef = self.card_defs[card_name]

        if cdef.is_spell:
            # Spells may be cast anywhere on the board (incl. the enemy side).
            pass
        else:
            # Troops must be placed on your side and not in the river or on towers.
            if sy in RIVER_ROWS:
                return False
            if team == P0 and sy >= RIVER_Y1:
                return False
            if team == P1 and sy <= RIVER_Y0:
                return False
            if self._tile_in_any_tower(sx, sy):
                return False

        if ps.elixir < cdef.cost:
            return False

        ps.elixir -= cdef.cost

        if cdef.is_spell:
            self._cast_spell(team, cdef, float(sx), float(sy))
        else:
            lane = 0 if sx < LANE_SPLIT_X else 1
            # Multi-spawn cards (e.g. Goblins) drop `count` units in a small cluster.
            for i in range(cdef.count):
                ox = (i - (cdef.count - 1) / 2.0) * 0.6
                px = clamp(sx + ox, 0.0, W_TILES - 1.0)
                self._spawn_unit(team, cdef.unit_def, px, float(sy), lane)

        # cycle card: replace used card in hand with next card in deck
        next_card = ps.deck[ps.deck_index % len(ps.deck)]
        ps.deck_index += 1
        ps.hand[hand_slot] = next_card
        return True

    def _cast_spell(self, team: int, cdef: CardDef, sx: float, sy: float):
        """Instant area damage to enemy units (and reduced damage to enemy towers)."""
        st = self.state
        enemy = self._enemy_team(team)
        for u in st.units.values():
            if u.team == enemy and u.hp > 0:
                if self._distance(u.x, u.y, sx, sy) <= cdef.spell_radius:
                    u.hp -= cdef.spell_damage
        if cdef.spell_tower_damage > 0:
            for tw in st.towers:
                if tw.team == enemy and tw.hp > 0:
                    tx, ty = tw.center
                    if self._distance(tx, ty, sx, sy) <= cdef.spell_radius:
                        tw.hp -= cdef.spell_tower_damage

    def _tile_in_any_tower(self, tx: int, ty: int) -> bool:
        for tw in self.state.towers:
            if tw.rect.collidepoint(tx, ty):
                return True
        return False

    def _spawn_unit(self, team: int, udef: UnitDef, x: float, y: float, lane: int):
        st = self.state
        uid = st.next_uid
        st.next_uid += 1
        st.units[uid] = Unit(uid=uid, team=team, udef=udef, x=x, y=y, hp=udef.max_hp, lane=lane, atk_cd=0.0)

    @staticmethod
    def _enemy_team(team: int) -> int:
        return P1 if team == P0 else P0

    @staticmethod
    def _lane_center_x(lane: int) -> int:
        return LEFT_LANE_CENTER_X if lane == 0 else RIGHT_LANE_CENTER_X

    def _nearest_enemy_tower(self, team: int, lane: int) -> Tower:
        st = self.state
        enemy = self._enemy_team(team)

        crown_kind = "crownL" if lane == 0 else "crownR"
        crown = next(t for t in st.towers if t.team == enemy and t.kind == crown_kind)
        king = next(t for t in st.towers if t.team == enemy and t.kind == "king")

        return crown if crown.hp > 0 else king

    @staticmethod
    def _distance(ax: float, ay: float, bx: float, by: float):
        return math.hypot(ax - bx, ay - by)

    def _update_units(self):
        st = self.state

        # how close to lane center to be considered "aligned"
        ALIGN_X_TOL = 0.25

        # Reduce attack cooldowns
        for u in st.units.values():
            u.atk_cd = max(0.0, u.atk_cd - DT)

        # Build lists for targeting
        units_by_team: dict[int, list[Unit]] = {P0: [], P1: []}
        for u in st.units.values():
            units_by_team[u.team].append(u)

        # For each unit, pick target within range or move
        dead_uids: list[int] = []
        for u in list(st.units.values()):
            if u.hp <= 0:
                dead_uids.append(u.uid)
                continue

            enemy_team = self._enemy_team(u.team)
            enemy_units = units_by_team[enemy_team]

            lane_x = float(self._lane_center_x(u.lane))
            aligned = abs(u.x - lane_x) <= ALIGN_X_TOL

            # Building-targeters (e.g. Giant) ignore enemy troops entirely and head
            # straight for the towers; everyone else engages the nearest enemy in range.
            if not u.udef.targets_buildings:
                in_range: list[tuple[float, Unit]] = []
                for v in enemy_units:
                    if v.hp <= 0:
                        continue
                    d = self._distance(v.x, v.y, u.x, u.y)
                    if d <= u.udef.attack_range_tiles:
                        in_range.append((d, v))
                in_range.sort(key=lambda t: t[0])

                if in_range:
                    if u.atk_cd <= 1e-6:
                        # attack closest unit if one exists
                        target: Unit = in_range[0][1]
                        target.hp -= u.udef.damage
                        u.atk_cd = 1.0 / u.udef.attacks_per_sec
                        continue
                    else:
                        continue

            # else, check for towers in range
            tw : Tower = self._nearest_enemy_tower(u.team, u.lane)
            tx, ty = tw.center
            distance = self._distance(tx, ty, u.x, u.y)

            if distance <= u.udef.attack_range_tiles:
                if u.atk_cd <= 1e-6:
                    tw.hp -= u.udef.damage
                    u.atk_cd = 1.0 / u.udef.attacks_per_sec
                    continue
                else:
                    continue

            # else, move toward the target tower via waypoints:
            #   1) if still on our side, slide to lane center (horizontal only),
            #   2) then cross the river straight down/up the bridge,
            #   3) once across, head directly (diagonally allowed) for the tower.
            # This keeps river crossings on the bridge without freezing movement
            # elsewhere, so off-center targets like the king tower stay reachable.
            across = (u.team == P0 and u.y > RIVER_Y1) or (u.team == P1 and u.y < RIVER_Y0)
            if not across:
                if not aligned:
                    goal_x, goal_y = lane_x, u.y # slide to lane center first
                else:
                    cross_y = RIVER_Y1 + 1 if u.team == P0 else RIVER_Y0 - 1
                    goal_x, goal_y = lane_x, cross_y # cross straight over the bridge
            else:
                goal_x, goal_y = tx, ty # head straight for the tower

            dx, dy = goal_x - u.x, goal_y - u.y
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                vx, vy = dx / dist, dy / dist
                step = u.udef.speed_tiles_per_sec * DT
                step = min(step, dist)

                new_x = u.x + vx * step
                new_y = u.y + vy * step

                u.x = clamp(new_x, 0, W_TILES)
                u.y = clamp(new_y, 0, H_TILES)

        # Cleanup dead units
        for uid in dead_uids:
            st.units.pop(uid, None)

    def _check_winner(self):
        st = self.state

        p0_king = next(t for t in st.towers if t.team == P0 and t.kind == "king")
        p1_king = next(t for t in st.towers if t.team == P1 and t.kind == "king")

        hp0, hp1 = p0_king.hp, p1_king.hp

        if hp0 <= 0 and hp1 <= 0:
            st.winner = None
        elif hp0 <= 0:
            st.winner = P1
        elif hp1 <= 0:
            st.winner = P0

# ====================
# Pygame Renderer
# ====================

class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("MiniClash")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(name=None, size=20)
        self.bigfont = pygame.font.SysFont(name=None, size=28)

    def draw(self, st: GameState):
        self.screen.fill(color=(22, 24, 28))

        # Board background
        board_rect = pygame.Rect(MARGIN_PX, MARGIN_PX, W_TILES * TILE_PX, H_TILES * TILE_PX)
        pygame.draw.rect(self.screen, (36, 40, 46), board_rect, border_radius=8)

        # Draw grid
        for x in range(W_TILES + 1):
            # vertical lines
            px = MARGIN_PX + x * TILE_PX
            pygame.draw.line(self.screen, (48, 52, 58), (px, MARGIN_PX), (px, MARGIN_PX + H_TILES * TILE_PX), 1)
        for y in range(H_TILES + 1):
            # horizontal lines
            py = MARGIN_PX + y * TILE_PX
            pygame.draw.line(self.screen, (48, 52, 58), (MARGIN_PX, py), (MARGIN_PX + W_TILES * TILE_PX, py), 1)

        # Draw river
        for ry in sorted(RIVER_ROWS):
            r = pygame.Rect(MARGIN_PX, MARGIN_PX + ry * TILE_PX, W_TILES * TILE_PX, TILE_PX)
            pygame.draw.rect(self.screen, (28, 60, 90), r)

        # Draw bridges
        plank_color = (140, 100, 60) # main wood
        plank_line_color = (110, 75, 40) # plank separators, darker than main wood
        edge_color = (60, 40, 30) # bridge border

        top_ry = min(RIVER_ROWS)
        height_tiles = len(RIVER_ROWS)

        for lane_center in (LEFT_LANE_CENTER_X, RIGHT_LANE_CENTER_X):
            shift = -1 if lane_center == LEFT_LANE_CENTER_X else 1
            left_tile = lane_center - BRIDGE_W_TILES // 2 + shift

            px = MARGIN_PX + left_tile * TILE_PX
            py = MARGIN_PX + top_ry * TILE_PX
            pw = BRIDGE_W_TILES * TILE_PX
            ph = height_tiles * TILE_PX
            bridge_rect = pygame.Rect(px, py, pw, ph)

            # Draw main bridge body
            pygame.draw.rect(self.screen, plank_color, bridge_rect, border_radius=6)

            # Draw plank separators
            num_planks = BRIDGE_W_TILES * 3
            for i in range(1, num_planks):
                lx = bridge_rect.x + int(i * bridge_rect.w / num_planks)
                pygame.draw.line(self.screen, plank_line_color, (lx, bridge_rect.y + 4), (lx, bridge_rect.y + bridge_rect.h - 4), width=2)

            pygame.draw.rect(self.screen, edge_color, bridge_rect, width=2, border_radius=6)

        # Draw towers
        for tw in st.towers:
            pxr = pygame.Rect(
                MARGIN_PX + tw.rect.x * TILE_PX,
                MARGIN_PX + tw.rect.y * TILE_PX,
                tw.rect.w * TILE_PX,
                tw.rect.h * TILE_PX)

            col = (190, 80, 80) if tw.team == P0 else (80, 140, 210)
            pygame.draw.rect(self.screen, col, pxr, border_radius=6)

            # HP bar
            hp_frac = 0.0 if tw.max_hp <= 0.0 else clamp(tw.hp / tw.max_hp, 0.0, 1.0)
            bar_h = 6
            bar = pygame.Rect(pxr.x, pxr.y - bar_h - 2, pxr.w, bar_h)
            pygame.draw.rect(self.screen, (20, 20, 20), bar)
            fill = pygame.Rect(pxr.x, pxr.y - bar_h - 2, int(hp_frac * pxr.w), bar_h)
            pygame.draw.rect(self.screen, (220, 220, 220), fill)

            label = self.font.render(tw.kind, True, (10, 10, 10))
            self.screen.blit(label, (pxr.x + 4, pxr.y + 4))

        # Draw units
        for u in st.units.values():
            cx, cy = tile_to_px(u.x, u.y)
            col = (230, 120, 120) if u.team == P0 else (120, 180, 240)
            pygame.draw.circle(self.screen, col, center=(cx, cy), radius=int(TILE_PX * 0.35))

            # unit hp bar
            hp_frac = clamp(u.hp / u.udef.max_hp, 0.0, 1.0)
            bar_w = int(TILE_PX * 0.8)
            bar_h = 5
            bar = pygame.Rect(cx - bar_w // 2, cy - int(TILE_PX * 0.55), bar_w, bar_h)
            pygame.draw.rect(self.screen, (20, 20, 20), bar)
            fill = pygame.Rect(bar.x, bar.y, bar_w * hp_frac, bar_h)
            pygame.draw.rect(self.screen, (240, 240, 240), fill)
            name_surf = self.font.render(u.udef.name, True, (10, 10, 10))
            name_rect = name_surf.get_rect(center=(cx, bar.y - 6))
            self.screen.blit(name_surf, name_rect)

        # UI
        ui_y = MARGIN_PX + H_TILES * TILE_PX + 10
        ui = pygame.Rect(MARGIN_PX, ui_y, W_TILES * TILE_PX, UI_HEIGHT_PX - 20)
        pygame.draw.rect(self.screen, (30, 32, 36), ui, border_radius=8)

        p0, p1 = st.players[P0], st.players[P1]
        ttxt = self.bigfont.render(f"t={st.time_left:5.1f}s tick={st.tick}", True, (230, 230, 230))
        self.screen.blit(ttxt, (MARGIN_PX + 12, ui_y + 8))

        p0txt = self.font.render(f"P0 elixir={p0.elixir:4.1f} hand={p0.hand}", True, (230, 180, 180))
        p1txt = self.font.render(f"P1 elixir={p1.elixir:4.1f} hand={p1.hand}", True, (180, 210, 240))
        self.screen.blit(p0txt, (MARGIN_PX + 12, ui_y + 38))
        self.screen.blit(p1txt, (MARGIN_PX + 12, ui_y + 58))

        if st.winner is not None:
            who = "P0 (top)" if st.winner == P0 else "P1 (bottom)"
            win = self.bigfont.render("Winner: " + who, True, (255, 255, 255))
            self.screen.blit(win, (MARGIN_PX + 12, MARGIN_PX + 12))

        pygame.display.flip()

    def tick(self):
        self.clock.tick(TPS)

    @staticmethod
    def close():
        pygame.quit()

def random_bot(env: MiniClash, team: int) -> tuple[int, int, int] | None:
    st = env.state
    ps = st.players[team]
    if ps.elixir < 3:
        return None
    if random.random() > 0.08:
        return None

    hand_slot = random.randrange(4)

    # pick random legal tile
    for _ in range(40):
        sx = random.randrange(W_TILES)
        if team == P0:
            sy = random.randrange(0, RIVER_Y0) # under river to bottom
        else:
            sy = random.randrange(RIVER_Y1, H_TILES) # 0 to mid

        if sy in RIVER_ROWS:
            continue
        if env._tile_in_any_tower(sx, sy):
            continue

        return hand_slot, sx, sy
    return None


def heuristic_bot(env: MiniClash, team: int) -> tuple[int, int, int] | None:
    """A simple rule-based opponent -- a much stronger baseline than random_bot.

    Priorities: (1) if the enemy is attacking on our side, defend the threatened lane,
    using Arrows on a swarm and a troop otherwise; (2) otherwise, once elixir is
    plentiful, start a push -- a Giant tank at the back if we have one, else a troop
    over the bridge. When neither applies, hold elixir (return None).
    """
    st = env.state
    ps = st.players[team]
    enemy = P1 if team == P0 else P0

    def affordable(name: str) -> bool:
        return ps.elixir >= env.card_defs[name].cost

    def find_slot(pred) -> int | None:
        for i, name in enumerate(ps.hand):
            if affordable(name) and pred(env.card_defs[name]):
                return i
        return None

    def lane_x(lane: int) -> int:
        return LEFT_LANE_CENTER_X if lane == 0 else RIGHT_LANE_CENTER_X

    on_my_half = (lambda y: y < RIVER_Y0) if team == P0 else (lambda y: y > RIVER_Y1)
    threats = [u for u in st.units.values()
               if u.team == enemy and u.hp > 0 and on_my_half(u.y)]

    if threats:
        # Focus the most-advanced threat's lane.
        threat = (min if team == P0 else max)(threats, key=lambda u: u.y)
        lane = threat.lane
        cluster = [u for u in threats if u.lane == lane]

        # A swarm of 3+ is worth an Arrows.
        if len(cluster) >= 3:
            slot = find_slot(lambda c: c.is_spell)
            if slot is not None:
                cx = sum(u.x for u in cluster) / len(cluster)
                cy = sum(u.y for u in cluster) / len(cluster)
                return slot, int(round(cx)), int(round(cy))

        # Otherwise meet it with a troop (not a building-targeter, not a spell).
        slot = find_slot(lambda c: c.unit_def is not None and not c.targets_buildings)
        if slot is not None:
            if team == P0:
                dy = int(clamp(threat.y, 7, RIVER_Y0 - 1))
            else:
                dy = int(clamp(threat.y, RIVER_Y1 + 1, H_TILES - 8))
            return slot, lane_x(lane), dy
        return None

    # No threats: build up, then push when we can afford a committed attack.
    if ps.elixir >= 9:
        push_lane = random.randrange(2)
        back_y = 8 if team == P0 else H_TILES - 8
        slot = find_slot(lambda c: c.targets_buildings)  # lead with a Giant if held
        if slot is not None:
            return slot, lane_x(push_lane), back_y
        slot = find_slot(lambda c: c.unit_def is not None)  # else any troop over the bridge
        if slot is not None:
            bridge_y = RIVER_Y0 - 1 if team == P0 else RIVER_Y1 + 1
            return slot, lane_x(push_lane), bridge_y
    return None


def main():
    env = MiniClash()
    ren = Renderer()

    running = True
    paused = False

    while running:
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    env.reset()

        if not paused:
            actions = {
                P0: heuristic_bot(env, P0),
                P1: heuristic_bot(env, P1)
            }
            env.step(actions)  # returns (state, {team: reward}, done)
        ren.draw(env.state)
        ren.tick()
    ren.close()

if __name__ == "__main__":
    main()





