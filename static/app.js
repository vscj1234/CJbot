class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
        };

        this.state = false;
        this.messages = [];
        this.inactivityTimer = null;
        this.inactivityPromptDisplayed = false;

        this.startInactivityTimer();
    }

    display() {
        const { openButton, chatBox, sendButton } = this.args;
        
        openButton.addEventListener('click', () => this.toggleState(chatBox));
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));
        
        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
            this.resetInactivityTimer();  // Reset the timer on typing
        });

        // Reset inactivity timer on any chatbox click or interaction
        chatBox.addEventListener('click', () => this.resetInactivityTimer());
        sendButton.addEventListener('click', () => this.resetInactivityTimer());
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // Show or hide the chatbox
        if (this.state) {
            chatbox.classList.add('chatbox--active');
            this.resetInactivityTimer();
        } else {
            chatbox.classList.remove('chatbox--active');
            this.clearInactivityTimer();
        }
    }

    onSendButton(chatbox) {
        const textField = chatbox.querySelector('input');
        const text1 = textField.value.trim();
        if (text1 === "") return;

        textField.value = "";  // Clear the input field

        let msg1 = { name: "User", message: text1 };
        this.messages.push(msg1);

        let typingMessage = { name: "CloudJune", message: "Typing..." };
        this.messages.push(typingMessage);
        this.updateChatText(chatbox);

        fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: text1,
                chat_history: this.messages
            }),
        })
        .then(response => response.json())
        .then(data => {
            // Remove "Typing..." message
            this.messages.pop();

            // Add bot's response to the messages array
            let botMessage = { name: "CloudJune", message: data.result };
            this.messages.push(botMessage);

            // Reset inactivity timer since the bot has replied
            this.resetInactivityTimer();

            // Update the chat UI
            this.updateChatText(chatbox);
        })
        .catch(error => {
            console.error('Error:', error);

            // Remove "Typing..." message and show error message
            this.messages.pop();
            let errorMessage = { name: "CloudJune", message: "Sorry, something went wrong." };
            this.messages.push(errorMessage);

            // Reset inactivity timer on error
            this.resetInactivityTimer();

            this.updateChatText(chatbox);
        });
    }

    updateChatText(chatbox) {
        let html = '';
        this.messages.slice().reverse().forEach(function(item) {
            if (item.name === "CloudJune") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>';
            } else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>';
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }

    startInactivityTimer() {
        this.inactivityTimer = setTimeout(() => {
            if (!this.inactivityPromptDisplayed) {
                this.showInactivityPrompt();
            }
        }, 180000);  // 3 minutes
    }

    resetInactivityTimer() {
        clearTimeout(this.inactivityTimer);  // Clear existing timer
        this.inactivityPromptDisplayed = false;
        this.startInactivityTimer();  // Start a new inactivity timer
    }

    clearInactivityTimer() {
        clearTimeout(this.inactivityTimer);
    }

    showInactivityPrompt() {
        this.inactivityPromptDisplayed = true;
        if (confirm("Do you want to continue the chat?")) {
            this.resetInactivityTimer();
        } else {
            this.terminateSession();
        }
    }

    terminateSession() {
        this.clearInactivityTimer();
        const chatbox = this.args.chatBox;
        let msg = { name: "CloudJune", message: "Bye! Have a great day." };
        this.messages.push(msg);
        this.updateChatText(chatbox);
        chatbox.classList.remove('chatbox--active');
    }
}

const chatbox = new Chatbox();
chatbox.display();
