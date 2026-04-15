from environment import UnoEnvironment
import numpy as np
from keras.models import load_model
import cv2
from pathlib import Path

# load gameplay model
gameplay_model = load_model('my_uno_model.h5')

# load card recognition model
rec_model = load_model('C:/Users/ebben/PycharmProjects/FinalAI.Uno/uno_model.keras')
with open('C:/Users/ebben/PycharmProjects/FinalAI.Uno/uno_labels.txt') as f:
    labels = [line.strip() for line in f if line.strip()]

import uno_config as cfg

env = UnoEnvironment(player_count=2) #cannot play w/ more w/ current rules

COLOURS = ['Red', 'Green', 'Blue', 'Yellow']
TYPES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Reverse', 'Draw2', 'Skip', 'Wild', 'Wild4']


def card_name(index):
    card = UnoEnvironment.CARD_TYPES[index]
    if card[0] is None:
        return TYPES[card[1]]
    return f"{COLOURS[card[0]]} {TYPES[card[1]]}"


def top_card_name():
    card = env.top_card
    colour = COLOURS[card[0]] if card[0] is not None else 'Any'
    return f"{colour} {TYPES[card[1]]}"


def name_to_index(inp):
    inp = inp.lower().replace("_", " ")

    for i in range(len(UnoEnvironment.CARD_TYPES)):
        if card_name(i).lower() == inp.lower():
            return i
    return None


def show_hand(player_index, label="Hand"):
    print(f"\n{label}:")
    for i, count in enumerate(env.players[player_index].cards):
        if count > 0:
            print(f"  {card_name(i)} (x{int(count)})")


def setup_hand(player_index, label):
    env.players[player_index].cards = np.zeros(len(UnoEnvironment.CARD_TYPES))
    print(f"\nEnter {label} cards one by one. Type 'done' when finished.")
    while True:
        inp = scan_or_type("AI Hand")
        if inp is None:
            continue
        inp = inp.strip()
        if inp.lower() == 'done':
            break
        # normalize input
        inp = inp.replace("_", " ")
        idx = name_to_index(inp)
        if idx is not None:
            env.players[player_index].cards[idx] += 1
            print(f"  Added {card_name(idx)}")
        else:
            print("  Card not recognised, try again")


def set_top_card():
    print("\nWhat is the top card on the pile?")
    #print("Type card name or press SPACE to scan with camera")

    result = scan_or_type("Top card")
    idx = name_to_index(result)
    if idx is not None:
        env.top_card = UnoEnvironment.CARD_TYPES[idx].copy()
        if env.top_card[0] is None:
            colour = input("What colour was called? (Red/Green/Blue/Yellow): ").strip()
            env.top_card[0] = COLOURS.index(colour.capitalize())
        print(f"  Top card set to {top_card_name()}")


def preprocess(frame_bgr):
    img = cv2.resize(frame_bgr, cfg.IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return np.expand_dims(img, 0)


def scan_card():
    """Open camera, press SPACE to scan, returns recognised card name"""
    cap = cv2.VideoCapture(cfg.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("  [Camera] SPACE to scan, Q to cancel")
    result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Scan card (SPACE to capture, Q to cancel)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            # take 5 frames and majority vote
            frames = [frame]
            for _ in range(4):
                ret, f = cap.read()
                if ret:
                    frames.append(f)

            predictions = []
            for f in frames:
                probs = rec_model.predict(preprocess(f), verbose=0)[0]
                predictions.append(int(np.argmax(probs)))

            smoothed = max(set(predictions), key=predictions.count)
            label = labels[smoothed] if smoothed < len(labels) else "unknown"
            confidence = float(rec_model.predict(preprocess(frames[-1]), verbose=0)[0][smoothed])
            cap.release()
            cv2.destroyAllWindows()


            print(f"  Recognised: {label} (confidence: {confidence:.0%})")

            # ask to confirm since model isn't perfect
            confirm = input(f"  Accept '{label}'? (y/n or type correction): ").strip()
            if confirm.lower() == 'y':
                result = label
                break
            elif confirm.lower() == 'n':
                print("  Try again...")
                return scan_card()
            else:
                result = confirm  # they typed the correct card
                break

    #cap.release()
    #cv2.destroyAllWindows()
    return result


def scan_or_type(prompt):
    """Either scan with camera or type manually"""
    print(f"\n{prompt} — type card name or press ENTER to scan with camera")
    inp = input("> ").strip()
    if inp == '':
        return scan_card()
    return inp


def ai_turn():
    env.turn = 0
    print("\n--- AI's turn ---")
    show_hand(0, "AI hand")
    print(f"Top card: {top_card_name()}")

    state = env.get_state()
    predictions = gameplay_model.predict(state.reshape(1, -1), verbose=0)[0]

    action = np.argmax([
        p if env.legal_move(i) else -1
        for i, p in enumerate(predictions)
    ])

    extra_turn = False

    if action < len(UnoEnvironment.CARD_TYPES):
        card = UnoEnvironment.CARD_TYPES[action].copy()
        card_type = TYPES[card[1]]

        print(f"==AI PLAYS: {card_name(action)}==")

        if card_type in ['Skip', 'Reverse']:
            print("  You lose your turn!")
            extra_turn = True

        elif card_type == 'Draw2':
            print("  You must draw 2 cards!")
            human_card_count[0] += 2
            extra_turn = True

        elif card_type == 'Wild4':
            print("  You must draw 4 cards!")
            human_card_count[0] += 4
            extra_turn = True
    else:
        print("  AI draws a card.")
        drawn = scan_or_type("What card did the AI draw?")
        if drawn:
            drawn = drawn.replace("_", " ")
            idx = name_to_index(drawn)
            if idx is not None:
                env.players[0].cards[idx] += 1
                print(f"  AI drew: {card_name(idx)}")

    env.step(action)
    #print(f"  Top card is now: {top_card_name()}")
    print(f"  AI has {env.players[0].num_cards()} cards left.")
    #print(f"  You have {human_card_count[0]} cards left.")
    return extra_turn


def human_turn():
    env.turn = 1
    print("\n--- Your turn ---")
    print(f"Top card: {top_card_name()}")
    print(f"You have {human_card_count[0]} cards.")
    print("What did you play? (press ENTER to scan, type card name, or type 'draw')")

    extra_turn = False

    while True:
        inp = input("> ").strip()

        if inp.lower() == 'draw':
            human_card_count[0] += 1
            env.step(len(UnoEnvironment.CARD_TYPES))
            print(f"  You drew a card. You now have {human_card_count[0]} cards.")
            return False

        if inp == '':
            inp = scan_card()
            if inp is None:
                continue

        inp = inp.replace("_", " ")
        idx = name_to_index(inp)

        if idx is None:
            print("  Card not recognised, try again.")
            continue

        card = UnoEnvironment.CARD_TYPES[idx].copy()
        card_type = TYPES[card[1]]

        # update game state
        env.top_card = card

        if env.top_card[0] is None:
            colour = input("What colour did you call? (Red/Green/Blue/Yellow): ").strip()
            env.top_card[0] = COLOURS.index(colour.capitalize())

        human_card_count[0] -= 1

        print(f"  Logged: you played {card_name(idx)}")
        #print(f"  Top card is now: {top_card_name()}")
        #print(f"  You have {human_card_count[0]} cards left.")

        env.turn = 0

        # --- EFFECTS (2-player UNO rules) ---
        if card_type in ['Skip', 'Reverse']:
            print("  AI loses its turn!")
            extra_turn = True

        elif card_type == 'Draw2':
            print("  AI must draw 2 cards!")
            for _ in range(2):
                drawn = scan_or_type("AI draws card:")
                didx = name_to_index(drawn)
                if didx is not None:
                    env.players[0].cards[didx] += 1
            extra_turn = True

        elif card_type == 'Wild4':
            print("  AI must draw 4 cards!")
            for _ in range(4):
                drawn = scan_or_type("AI draws card:")
                didx = name_to_index(drawn)
                if didx is not None:
                    env.players[0].cards[didx] += 1
            extra_turn = True

        return extra_turn


print("=== UNO AI ===\n")

set_top_card()
setup_hand(0, "AI's")

ai_card_count = env.players[0].num_cards()
human_card_count = [ai_card_count]
print(f"\nYou start with {human_card_count[0]} cards (same as AI).")

turn = 1  # 1 = you, 0 = AI

while True:
    print(f"\n{'=' * 30}")

    if turn == 0:
        env.turn = 0
        extra_turn = ai_turn()

        if env.players[0].num_cards() == 0:
            print("\n" + "=" * 30)
            print("  AI WINS! Better luck next time!")
            print("=" * 30)
            break

    else:
        env.turn = 1
        extra_turn = human_turn()

        if human_card_count[0] == 0:
            print("\n" + "=" * 30)
            print("  YOU WIN! Congratulations!")
            print("=" * 30)
            break

    #KEY LOGIC
    if not extra_turn:
        turn = 1 - turn