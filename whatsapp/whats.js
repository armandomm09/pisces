const { Client, LocalAuth, MessageMedia } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const { group } = require('console');
const { response } = require('express');


let clientReady = false;
const client = new Client({
    authStrategy: new LocalAuth()
});

// Cuando el cliente esté listo
client.once('ready', async () => {
    console.log('El cliente de WhatsApp está listo.');
    clientReady = true;

});

client.on('qr', (qr) => {
    console.log('Escanea este código QR con tu aplicación de WhatsApp:');
    qrcode.generate(qr, { small: true });
});

client.on('message', async message => {
    console.log(message.from)
    let messageBool = message.from === "5212223201384@c.us" || message.from === "5212228418062@c.us" || message.from === "120363041061664592@g.us" || message.from === "120363041061664592@g.us" || message.from === "5212214239597@c.us"
    if ((messageBool) && message.type === "image") {

        const imageMedia = await message.downloadMedia()
        const base64Data = imageMedia.data;

        const byteCharacters = atob(base64Data);
        const byteNumbers = Array.from(byteCharacters).map(char => char.charCodeAt(0));
        const byteArray = new Uint8Array(byteNumbers);

        const blob = new Blob([byteArray], { type: imageMedia.mimetype });

        const file = new File([blob], "downloadedImage.jpg", { type: imageMedia.mimetype });

        console.log(file);
        let formData = new FormData()
        formData.append("file", file)

        await fetch("http://localhost:8000/image", {
            method: 'POST',
            body: formData
        }).then(async response => {
            if (response.ok) {
                const result = await response.json()
                console.log(response.status)
                console.log("Filename is", result.filename);
                console.log("Fish info:\n", result.fish_info)
                let imageResponse = await MessageMedia.fromUrl(`http://localhost:8000/fetch_image/${result.filename}`)
                let imageCaption = `Parece que capturaste un\n${result.fish_info[0].name} 😨😨😨\nMejor conocido como ${result.nickname}`
                await message.reply(imageResponse, message.from, {caption: imageCaption})
                // await message.reply(imageResponse, message.from)
                
            } else {
                console.error(`HTTP Error: ${response.status} - ${response.statusText}`);
            }
        }).catch(error => {
            console.log("Error at: ", error)
        })
    }
})


client.initialize();

