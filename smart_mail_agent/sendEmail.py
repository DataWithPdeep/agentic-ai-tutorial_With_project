import smtplib
from email.message import EmailMessage


def send_mail(recipient_email, Subject, body):    
    # Define email details
    sender_email = "psrana344@gmail.com"
    sender_password = "titr juty gdes osfy"  

    # Create the message
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = Subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    # Connect to the SMTP server and send
    try:
        # Use port 465 for SMTP_SSL, or port 587 for STARTTLS
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()  # Upgrade the insecure connection to TLS
            server.ehlo()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return "Email sent successfully!"
    except Exception as e:
        return f"Failed to send email: {e}"
