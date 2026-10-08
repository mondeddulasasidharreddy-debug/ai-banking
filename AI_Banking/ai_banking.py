"""
AI Banking - a simple base project
-----------------------------------
Features:
  1. Core banking: create account, login (PIN), deposit, withdraw, transfer, history
  2. AI Fraud Detection: statistical (z-score) + rule-based anomaly checks
  3. AI Assistant: keyword/intent chatbot that understands plain-English requests
  4. Data saved to bank_data.json

Run:  python ai_banking.py
"""

import hashlib
import json
import os
import random
import re
import statistics
import time
from datetime import datetime

DATA_FILE = "bank_data.json"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def hash_pin(pin: str, salt: str) -> str:
    return hashlib.sha256((salt + pin).encode()).hexdigest()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------------
# AI Module 1: Fraud Detection
# ----------------------------------------------------------------------------
class FraudDetector:
    """Scores a transaction from 0 (safe) to 100 (very suspicious)."""

    LARGE_AMOUNT = 50000       # absolute limit that is always suspicious
    MIN_HISTORY = 5            # transactions needed before using statistics
    RAPID_WINDOW_SEC = 60      # window for the "too many transactions" rule
    RAPID_LIMIT = 3

    def score(self, amount: float, history: list):
        reasons, risk = [], 0

        # Rule 1: very large amount
        if amount >= self.LARGE_AMOUNT:
            risk += 40
            reasons.append(f"Amount is very large (>= {self.LARGE_AMOUNT:,})")

        # Rule 2: statistical outlier vs. the customer's own behaviour (z-score)
        amounts = [t["amount"] for t in history if t["type"] in ("withdraw", "transfer_out")]
        if len(amounts) >= self.MIN_HISTORY:
            mean = statistics.mean(amounts)
            stdev = statistics.pstdev(amounts) or 1.0
            z = (amount - mean) / stdev
            if z > 3:
                risk += 40
                reasons.append(f"Amount is {z:.1f} std-devs above your usual spending")
            elif z > 2:
                risk += 20
                reasons.append("Amount is higher than your usual spending")

        # Rule 3: many transactions in a short time (velocity check)
        recent = [t for t in history if time.time() - t["ts"] <= self.RAPID_WINDOW_SEC]
        if len(recent) >= self.RAPID_LIMIT:
            risk += 30
            reasons.append("Too many transactions in a short time")

        # Rule 4: unusual hour (midnight - 5 AM)
        if datetime.now().hour < 5:
            risk += 10
            reasons.append("Transaction at an unusual hour")

        return min(risk, 100), reasons

    def is_suspicious(self, amount, history):
        risk, reasons = self.score(amount, history)
        return risk >= 40, risk, reasons


# ----------------------------------------------------------------------------
# Core Banking
# ----------------------------------------------------------------------------
class Bank:
    def __init__(self):
        self.accounts = {}
        self.fraud = FraudDetector()
        self.load()

    # ---- persistence ----
    def load(self):
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE) as f:
                self.accounts = json.load(f)

    def save(self):
        with open(DATA_FILE, "w") as f:
            json.dump(self.accounts, f, indent=2)

    # ---- accounts ----
    def create_account(self, name: str, pin: str, opening: float = 0.0) -> str:
        acc_no = str(random.randint(10**9, 10**10 - 1))
        while acc_no in self.accounts:
            acc_no = str(random.randint(10**9, 10**10 - 1))
        salt = os.urandom(8).hex()
        self.accounts[acc_no] = {
            "name": name,
            "salt": salt,
            "pin_hash": hash_pin(pin, salt),
            "balance": float(opening),
            "transactions": [],
        }
        if opening > 0:
            self._log(acc_no, "deposit", opening, "Opening deposit")
        self.save()
        return acc_no

    def login(self, acc_no: str, pin: str) -> bool:
        acc = self.accounts.get(acc_no)
        return bool(acc) and acc["pin_hash"] == hash_pin(pin, acc["salt"])

    def _log(self, acc_no, ttype, amount, note=""):
        acc = self.accounts[acc_no]
        acc["transactions"].append({
            "type": ttype, "amount": amount, "note": note,
            "balance_after": acc["balance"], "time": now_str(), "ts": time.time(),
        })

    # ---- operations ----
    def deposit(self, acc_no, amount):
        if amount <= 0:
            return False, "Amount must be positive."
        self.accounts[acc_no]["balance"] += amount
        self._log(acc_no, "deposit", amount)
        self.save()
        return True, f"Deposited {amount:,.2f}. New balance: {self.accounts[acc_no]['balance']:,.2f}"

    def withdraw(self, acc_no, amount, confirm_fn=None):
        acc = self.accounts[acc_no]
        if amount <= 0:
            return False, "Amount must be positive."
        if amount > acc["balance"]:
            return False, "Insufficient balance."
        blocked = self._fraud_gate(acc, amount, confirm_fn)
        if blocked:
            return False, blocked
        acc["balance"] -= amount
        self._log(acc_no, "withdraw", amount)
        self.save()
        return True, f"Withdrew {amount:,.2f}. New balance: {acc['balance']:,.2f}"

    def transfer(self, acc_no, to_acc, amount, confirm_fn=None):
        if to_acc not in self.accounts or to_acc == acc_no:
            return False, "Invalid destination account."
        acc = self.accounts[acc_no]
        if amount <= 0:
            return False, "Amount must be positive."
        if amount > acc["balance"]:
            return False, "Insufficient balance."
        blocked = self._fraud_gate(acc, amount, confirm_fn)
        if blocked:
            return False, blocked
        acc["balance"] -= amount
        self.accounts[to_acc]["balance"] += amount
        self._log(acc_no, "transfer_out", amount, f"To {to_acc}")
        self._log(to_acc, "transfer_in", amount, f"From {acc_no}")
        self.save()
        return True, f"Transferred {amount:,.2f} to {to_acc}. New balance: {acc['balance']:,.2f}"

    def _fraud_gate(self, acc, amount, confirm_fn):
        """Returns an error message if the transaction is blocked, else None."""
        suspicious, risk, reasons = self.fraud.is_suspicious(amount, acc["transactions"])
        if not suspicious:
            return None
        print(f"\n  FRAUD ALERT (risk {risk}/100):")
        for r in reasons:
            print(f"    - {r}")
        if confirm_fn and confirm_fn("  Do you still want to proceed? (yes/no): "):
            return None
        return "Transaction cancelled by fraud protection."

    def history(self, acc_no, n=10):
        return self.accounts[acc_no]["transactions"][-n:]


# ----------------------------------------------------------------------------
# AI Module 2: Assistant (intent detection)
# ----------------------------------------------------------------------------
class BankAssistant:
    """Tiny NLP assistant: scores keywords per intent and picks the best one."""

    INTENTS = {
        "balance":  ["balance", "how much", "money", "funds", "have"],
        "history":  ["history", "transactions", "statement", "recent", "last"],
        "deposit":  ["deposit", "add", "credit", "put"],
        "withdraw": ["withdraw", "take out", "cash", "debit"],
        "transfer": ["transfer", "send", "pay"],
        "insights": ["spend", "spending", "insight", "analysis", "summary", "budget"],
        "help":     ["help", "what can you do", "options", "commands"],
    }

    def __init__(self, bank: Bank, acc_no: str):
        self.bank, self.acc_no = bank, acc_no

    def detect_intent(self, text: str):
        text = text.lower()
        scores = {i: sum(1 for k in kws if k in text) for i, kws in self.INTENTS.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else None

    @staticmethod
    def extract_amount(text: str):
        m = re.search(r"(\d+(?:,\d{3})*(?:\.\d+)?)", text)
        return float(m.group(1).replace(",", "")) if m else None

    @staticmethod
    def extract_account(text: str):
        m = re.search(r"\b(\d{10})\b", text)
        return m.group(1) if m else None

    def spending_insights(self):
        tx = self.bank.accounts[self.acc_no]["transactions"]
        spent = [t["amount"] for t in tx if t["type"] in ("withdraw", "transfer_out")]
        earned = [t["amount"] for t in tx if t["type"] in ("deposit", "transfer_in")]
        if not spent:
            return "No spending recorded yet."
        msg = (f"You spent {sum(spent):,.2f} across {len(spent)} transactions "
               f"(average {statistics.mean(spent):,.2f}, largest {max(spent):,.2f}). "
               f"Total money in: {sum(earned):,.2f}.")
        if sum(earned) and sum(spent) / sum(earned) > 0.8:
            msg += " Tip: you're spending over 80% of your income - consider saving more."
        return msg

    def reply(self, text: str, confirm_fn=None) -> str:
        intent = self.detect_intent(text)
        acc = self.bank.accounts[self.acc_no]
        amount = self.extract_amount(text)

        if intent == "balance":
            return f"Your balance is {acc['balance']:,.2f}."
        if intent == "history":
            tx = self.bank.history(self.acc_no, 5)
            if not tx:
                return "No transactions yet."
            return "Last transactions:\n" + "\n".join(
                f"  {t['time']}  {t['type']:<13}{t['amount']:>12,.2f}" for t in tx)
        if intent == "insights":
            return self.spending_insights()
        if intent == "deposit":
            return self.bank.deposit(self.acc_no, amount)[1] if amount else "How much to deposit?"
        if intent == "withdraw":
            return self.bank.withdraw(self.acc_no, amount, confirm_fn)[1] if amount else "How much to withdraw?"
        if intent == "transfer":
            to = self.extract_account(text)
            if not amount or not to:
                return "Say e.g. 'transfer 500 to 1234567890'."
            return self.bank.transfer(self.acc_no, to, amount, confirm_fn)[1]
        if intent == "help":
            return ("I can: check balance, show history, deposit, withdraw, transfer, "
                    "and give spending insights. Try 'withdraw 200' or 'send 500 to 1234567890'.")
        return "Sorry, I didn't understand. Type 'help' to see what I can do."


# ----------------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------------
def ask_yes(prompt: str) -> bool:
    return input(prompt).strip().lower() in ("y", "yes")


def ask_float(prompt: str):
    try:
        return float(input(prompt))
    except ValueError:
        print("Invalid number.")
        return None


def user_menu(bank: Bank, acc_no: str):
    assistant = BankAssistant(bank, acc_no)
    while True:
        acc = bank.accounts[acc_no]
        print(f"\n--- Welcome, {acc['name']} (Balance: {acc['balance']:,.2f}) ---")
        print("1. Deposit\n2. Withdraw\n3. Transfer\n4. History\n5. Spending insights\n"
              "6. Chat with AI assistant\n7. Logout")
        choice = input("Choose: ").strip()

        if choice == "1":
            amt = ask_float("Amount: ")
            if amt is not None:
                print(bank.deposit(acc_no, amt)[1])
        elif choice == "2":
            amt = ask_float("Amount: ")
            if amt is not None:
                print(bank.withdraw(acc_no, amt, ask_yes)[1])
        elif choice == "3":
            to = input("Destination account no.: ").strip()
            amt = ask_float("Amount: ")
            if amt is not None:
                print(bank.transfer(acc_no, to, amt, ask_yes)[1])
        elif choice == "4":
            for t in bank.history(acc_no):
                print(f"  {t['time']}  {t['type']:<13}{t['amount']:>12,.2f}  {t['note']}")
        elif choice == "5":
            print(assistant.spending_insights())
        elif choice == "6":
            print("AI Assistant ready. Type 'exit' to leave the chat.")
            while True:
                text = input("You: ").strip()
                if text.lower() in ("exit", "quit"):
                    break
                print("Bot:", assistant.reply(text, ask_yes))
        elif choice == "7":
            break
        else:
            print("Invalid choice.")


def main():
    bank = Bank()
    print("=== AI BANKING ===")
    while True:
        print("\n1. Create account\n2. Login\n3. Exit")
        choice = input("Choose: ").strip()
        if choice == "1":
            name = input("Full name: ").strip()
            pin = input("Set 4-digit PIN: ").strip()
            if not (pin.isdigit() and len(pin) == 4) or not name:
                print("Name required and PIN must be exactly 4 digits.")
                continue
            opening = ask_float("Opening deposit (0 for none): ") or 0.0
            print("Account created! Your account number:", bank.create_account(name, pin, opening))
        elif choice == "2":
            acc_no = input("Account number: ").strip()
            pin = input("PIN: ").strip()
            if bank.login(acc_no, pin):
                user_menu(bank, acc_no)
            else:
                print("Invalid account number or PIN.")
        elif choice == "3":
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
