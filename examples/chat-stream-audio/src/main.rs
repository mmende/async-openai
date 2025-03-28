use std::error::Error;
use std::io::{stdout, Write};

use async_openai::types::{
    ChatCompletionAudio, ChatCompletionAudioFormat, ChatCompletionAudioVoice,
    ChatCompletionModalities, ChatCompletionRequestUserMessageArgs,
};
use async_openai::{types::CreateChatCompletionRequestArgs, Client};
use base64::prelude::*;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = Client::new();

    let modalities = vec![
        ChatCompletionModalities::Text,
        ChatCompletionModalities::Audio,
    ];
    let audio = ChatCompletionAudio {
        voice: ChatCompletionAudioVoice::Alloy,
        format: ChatCompletionAudioFormat::Pcm16,
    };

    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o-audio-preview")
        .max_tokens(2048u32)
        .modalities(modalities)
        .audio(audio)
        .messages([ChatCompletionRequestUserMessageArgs::default()
            .content("Give a very short introduction speech in one sentence about the Rust library async-openai")
            .build()?
            .into()])
        .build()?;

    let mut stream = client.chat().create_stream(request).await?;

    // Create a sox play process that will play the audio stream
    let mut play_process = std::process::Command::new("play")
        .arg("-q")
        .arg("-t")
        .arg("raw")
        .arg("-e")
        .arg("signed-integer")
        .arg("-b")
        .arg("16")
        .arg("-c")
        .arg("1")
        .arg("-r")
        .arg("24000")
        .arg("-")
        .stdin(std::process::Stdio::piped())
        .spawn()
        .expect("Failed to start play process");
    let sox_stdin = play_process
        .stdin
        .as_mut()
        .expect("Failed to open sox stdin");

    // From Rust docs on print: https://doc.rust-lang.org/std/macro.print.html
    //
    //  Note that stdout is frequently line-buffered by default so it may be necessary
    //  to use io::stdout().flush() to ensure the output is emitted immediately.
    //
    //  The print! macro will lock the standard output on each call.
    //  If you call print! within a hot loop, this behavior may be the bottleneck of the loop.
    //  To avoid this, lock stdout with io::stdout().lock():

    let mut lock = stdout().lock();
    while let Some(result) = stream.next().await {
        match result {
            Ok(response) => {
                match response.choices.first() {
                    Some(choice) => {
                        if let Some(ref audio) = choice.delta.audio {
                            if let Some(ref transcript) = audio.transcript {
                                // Print the transcription to stdout
                                write!(lock, "{}", transcript).unwrap();
                            }
                            if let Some(ref audio_data) = audio.data {
                                // Convert base64 encoded audio data to bytes
                                let audio_data = BASE64_STANDARD.decode(audio_data)?;
                                // Write the audio data to the sox stdin
                                if let Err(err) = sox_stdin.write_all(&audio_data) {
                                    writeln!(lock, "error: {err}").unwrap();
                                }
                            }
                        }
                    }
                    None => {}
                }
            }
            Err(err) => {
                writeln!(lock, "error: {err}").unwrap();
            }
        }
        stdout().flush()?;
    }

    // Close the sox stdin to signal the end of the audio stream
    if let Err(err) = sox_stdin.flush() {
        writeln!(lock, "error: {err}").unwrap();
    }
    if let Err(err) = play_process.wait() {
        writeln!(lock, "error: {err}").unwrap();
    }
    // Check if the play process exited successfully
    if !play_process.try_wait()?.unwrap().success() {
        writeln!(lock, "error: play process exited with error").unwrap();
    }

    Ok(())
}
