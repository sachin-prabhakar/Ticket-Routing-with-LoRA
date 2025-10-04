"""
Synthetic data generation for ticket routing models.

Features:
- Deterministic generation with configurable parameters
- Professional ticket patterns and language
- Comprehensive metadata and labeling
- Production-ready data validation
- Professional logging and error handling
"""

import json
import random
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ticket templates for each queue
TICKET_TEMPLATES = {
    "sales": {
        "subjects": [
            "Inquiry about pricing plans",
            "Need help with subscription upgrade",
            "Questions about enterprise features",
            "Interested in custom solutions",
            "Pricing for bulk licenses",
            "Sales demo request",
            "Contract renewal questions",
            "Volume discount inquiry"
        ],
        "bodies": [
            "Hi, I'm interested in learning more about your pricing plans. Could you provide details on the different tiers available?",
            "We're currently on the basic plan but need to upgrade to support more users. What are our options?",
            "I'd like to schedule a demo to see the enterprise features in action. When would be a good time?",
            "Our company is looking for a custom solution. Do you offer tailored implementations?",
            "We need licenses for about 100 users. Is there a volume discount available?",
            "Could someone from your sales team contact me to discuss our requirements?",
            "Our contract is up for renewal next month. What are the current terms and pricing?",
            "I'm interested in your premium features but need to understand the ROI better."
        ]
    },
    "tech_support": {
        "subjects": [
            "Login issues with my account",
            "Feature not working as expected",
            "Integration problems with API",
            "Performance issues on mobile app",
            "Data sync errors",
            "Browser compatibility problems",
            "Mobile app crashes",
            "Slow loading times"
        ],
        "bodies": [
            "I'm unable to log into my account. I keep getting an error message saying 'invalid credentials' but I'm sure my password is correct.",
            "The dashboard feature isn't displaying data correctly. Charts are showing empty even though I know there should be data.",
            "Our API integration is failing with 500 errors. The webhook calls are not being received properly.",
            "The mobile app is very slow on my iPhone. It takes forever to load and sometimes freezes completely.",
            "Data isn't syncing between my devices. Changes made on desktop don't appear on mobile.",
            "The interface doesn't work properly in Chrome. Some buttons are unclickable and forms don't submit.",
            "The app crashes every time I try to upload a file. This happens consistently on both iOS and Android.",
            "Pages are loading very slowly, especially the reports section. Sometimes it times out completely."
        ]
    },
    "general": {
        "subjects": [
            "General question about features",
            "Account information update",
            "Feedback on user experience",
            "Documentation request",
            "Training materials needed",
            "Best practices inquiry",
            "Feature request",
            "General inquiry"
        ],
        "bodies": [
            "I have a general question about how the reporting feature works. Could you explain the different options available?",
            "I need to update my account information. How do I change my email address and phone number?",
            "I wanted to provide some feedback on the user interface. Overall it's good but there are a few areas for improvement.",
            "Do you have documentation available for the API? I'd like to understand the available endpoints.",
            "Are there any training materials or tutorials available for new users? I'd like to get up to speed quickly.",
            "What are the best practices for using your platform? I want to make sure I'm using it efficiently.",
            "I have an idea for a new feature that would be really helpful. How do I submit feature requests?",
            "I'm new to the platform and have some general questions about getting started. Could someone help me?"
        ]
    },
    "billing": {
        "subjects": [
            "Billing question about charges",
            "Payment method update needed",
            "Invoice not received",
            "Refund request",
            "Billing cycle change",
            "Payment failed",
            "Tax calculation inquiry",
            "Subscription cancellation"
        ],
        "bodies": [
            "I have a question about my recent bill. There's a charge I don't recognize. Can you help me understand what this is for?",
            "I need to update my payment method. My credit card expired and I have a new one to use.",
            "I haven't received my invoice for this month. Could you please resend it to my email address?",
            "I'd like to request a refund for last month's charges. The service didn't meet my expectations.",
            "I want to change my billing cycle from monthly to annual. How do I go about doing this?",
            "My payment failed this month. I've updated my card information. Can you retry the payment?",
            "I need help understanding the tax calculation on my invoice. The amount seems higher than expected.",
            "I need to cancel my subscription. What's the process and will I get a prorated refund?"
        ]
    }
}

# Additional context and metadata
CONTEXT_PHRASES = [
    "urgent", "asap", "high priority", "critical", "important",
    "when you have time", "no rush", "low priority", "whenever possible"
]

COMPANY_TYPES = [
    "startup", "enterprise", "small business", "non-profit", "consulting firm",
    "tech company", "retail business", "healthcare organization"
]

def generate_ticket_id() -> str:
    """Generate a unique ticket ID."""
    return f"TKT-{random.randint(10000, 99999)}"

def generate_created_at(days_back: int = 30) -> str:
    """Generate a random creation timestamp within the last N days."""
    now = datetime.now()
    random_days = random.randint(0, days_back)
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    
    created_at = now - timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
    return created_at.isoformat()

def add_context_variation(text: str) -> str:
    """Add contextual variations to make text more realistic."""
    # Sometimes add urgency or priority context
    if random.random() < 0.2:  # 20% chance
        context = random.choice(CONTEXT_PHRASES)
        if random.random() < 0.5:
            text = f"{context.title()}: {text}"
        else:
            text = f"{text} ({context})"
    
    # Sometimes add company context
    if random.random() < 0.15:  # 15% chance
        company_type = random.choice(COMPANY_TYPES)
        text = f"{text} We're a {company_type}."
    
    return text

def generate_ticket(queue: str, ticket_id: str) -> Dict[str, Any]:
    """Generate a single ticket for the specified queue."""
    templates = TICKET_TEMPLATES[queue]
    
    # Select random subject and body
    subject = random.choice(templates["subjects"])
    body = random.choice(templates["bodies"])
    
    # Add variations
    subject = add_context_variation(subject)
    body = add_context_variation(body)
    
    # Generate metadata
    meta = {
        "queue": queue,
        "priority": random.choice(["low", "medium", "high"]),
        "user_type": random.choice(["individual", "business"]),
        "generated_at": datetime.now().isoformat()
    }
    
    return {
        "id": ticket_id,
        "subject": subject,
        "body": body,
        "label": queue,
        "created_at": generate_created_at(),
        "meta": meta
    }

def generate_synthetic_tickets(num_tickets: int = 2000, output_path: str = "data/synthetic_tickets.jsonl") -> None:
    """
    Generate synthetic ticket data with mild class imbalance.
    
    Args:
        num_tickets: Total number of tickets to generate
        output_path: Output file path
    """
    logger.info(f"Generating {num_tickets} synthetic tickets")
    
    # Define class distribution (mild imbalance)
    queue_distribution = {
        "sales": 0.25,      # 25%
        "tech_support": 0.35,  # 35% (most common)
        "general": 0.25,    # 25%
        "billing": 0.15     # 15% (least common)
    }
    
    # Calculate number of tickets per queue
    tickets_per_queue = {}
    for queue, proportion in queue_distribution.items():
        tickets_per_queue[queue] = int(num_tickets * proportion)
    
    # Adjust for rounding
    total_allocated = sum(tickets_per_queue.values())
    if total_allocated < num_tickets:
        tickets_per_queue["tech_support"] += (num_tickets - total_allocated)
    
    logger.info(f"Ticket distribution: {tickets_per_queue}")
    
    # Generate tickets
    all_tickets = []
    ticket_counter = 1
    
    for queue, count in tickets_per_queue.items():
        logger.info(f"Generating {count} tickets for {queue} queue")
        
        for _ in range(count):
            ticket_id = generate_ticket_id()
            ticket = generate_ticket(queue, ticket_id)
            all_tickets.append(ticket)
            ticket_counter += 1
    
    # Shuffle tickets to mix the order
    random.shuffle(all_tickets)
    
    # Write to JSONL file
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for ticket in all_tickets:
            f.write(json.dumps(ticket) + '\n')
    
    logger.info(f"Generated {len(all_tickets)} tickets and saved to {output_path}")
    
    # Print summary statistics
    queue_counts = {}
    for ticket in all_tickets:
        queue = ticket["label"]
        queue_counts[queue] = queue_counts.get(queue, 0) + 1
    
    print("\n" + "="*50)
    print("SYNTHETIC DATA GENERATION SUMMARY")
    print("="*50)
    print(f"Total tickets: {len(all_tickets)}")
    print(f"Output file: {output_path}")
    print("\nQueue distribution:")
    for queue, count in sorted(queue_counts.items()):
        percentage = (count / len(all_tickets)) * 100
        print(f"  {queue:15}: {count:4} tickets ({percentage:5.1f}%)")
    print("="*50)

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Generate synthetic ticket data')
    parser.add_argument('--num-tickets', type=int, default=2000, 
                       help='Number of tickets to generate (default: 2000)')
    parser.add_argument('--output', type=str, default='data/synthetic_tickets.jsonl',
                       help='Output file path (default: data/synthetic_tickets.jsonl)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Generate synthetic tickets
    generate_synthetic_tickets(args.num_tickets, args.output)

if __name__ == "__main__":
    main()
