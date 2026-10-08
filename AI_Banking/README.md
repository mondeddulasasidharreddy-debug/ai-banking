# AI Banking

A simple command-line banking system with two AI features.

## Features
- Create account, PIN login (salted SHA-256 hash)
- Deposit, withdraw, transfer, transaction history
- AI fraud detection (z-score + rule-based risk score, 0-100)
- AI assistant chatbot (plain-English commands, spending insights)
- Data saved to bank_data.json

## Run
    python ai_banking.py

Requires Python 3.8+. No extra packages needed.

## Try the assistant
    withdraw 200
    what's my balance
    send 500 to 1234567890
    show my spending insights
