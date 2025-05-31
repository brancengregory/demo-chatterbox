# Ensure 'here' and 'cli' are installed:
# install.packages(c("here", "cli", "reticulate"))

library(reticulate)
library(here)
library(cli)

# --- 1. Setup and Python Initialization ---
cli_h1("Setup and Python Initialization")

# Optional: Specify your Python environment
# tryCatch({
#   # Replace "my_chatterbox_env" with the name of your Conda env or path to venv
#   # use_condaenv("my_chatterbox_env", required = TRUE)
#   # use_virtualenv("path/to/your/venv", required = TRUE)
#   cli_alert_success("Using specified Python environment.")
# }, error = function(e) {
#   cli_alert_warning("Could not set specified Python environment. Using default. Error: {e$message}")
# })

py_versions_supported <- ">=3.12,<3.13"
cli_alert_info("Requiring Python version: {py_versions_supported}")

tryCatch({
  py_require(
    c(
      "chatterbox-tts",
      "torchaudio",
      "torch" # For device checks and cache clearing
    ),
    python_version = py_versions_supported
  )
  cli_alert_success("Python dependencies met.")
}, error = function(e) {
  cli_abort("Failed to meet Python requirements. Please check your Python environment and installed packages. Error: {e$message}")
})


# Import Python modules
cb <- NULL
ta <- NULL
torch <- NULL
py_gc <- NULL

tryCatch({
  cli_process_start("Importing Python modules")
  cb <- import("chatterbox.tts")
  ta <- import("torchaudio")
  torch <- import("torch")
  py_gc <- import("gc") # Python's garbage collector
  cli_process_done()
  cli_alert_success("Python modules imported.")
}, error = function(e) {
  cli_abort("Failed to import required Python modules. Error: {e$message}")
})


# --- 2. Core Function Definitions ---
cli_h1("Core Function Definitions")

init_chatterbox <- function(device_preference = "cuda") {
  selected_device <- device_preference
  process_msg <- "Initializing ChatterboxTTS model on device: {.val {selected_device}}"
  cli_process_start(process_msg, .envir = environment())

  if (selected_device == "cuda" && (is.null(torch) || !torch$cuda$is_available())) {
    cli_alert_warning("CUDA device preferred but not available or torch not imported. Falling back to CPU.")
    selected_device <- "cpu"
    # Update message for process done
    process_msg <- "Initializing ChatterboxTTS model on device: {.val {selected_device}} (fallback)"
  }

  model <- cb$ChatterboxTTS$from_pretrained(device = selected_device)
  cli_process_done() # This will use the updated process_msg if fallback occurred
  cli_alert_info("ChatterboxTTS model initialized on {.val {selected_device}}.")
  return(model)
}

chatterbox <- function(text_to_speak, output_file_path, current_model, audio_prompt_file_path = NULL) {
  if (is.null(current_model)) {
    cli_abort("Model not provided to chatterbox function.")
  }

  text_to_speak <- readChar(text_to_speak, 1000)

  cli_process_start("Generating audio for: {.file {basename(output_file_path)}}")
  generated_wav <- current_model$generate(
    text_to_speak,
    audio_prompt_path = audio_prompt_file_path
  )
  cli_process_done()

  cli_process_start("Saving audio to {.path {output_file_path}}")
  tryCatch({
    ta$save(
      output_file_path,
      generated_wav,
      as.integer(current_model$sr) # Ensure sample rate is an integer
    )
    cli_process_done()
  }, error = function(e) {
    cli_process_failed()
    cli_alert_danger("Failed to save audio: {e$message}")
  })

  # Clean up Python objects and CUDA cache
  # Remove R's reference. Python object itself will be handled by py_gc.
  rm(generated_wav)

  cli_process_start("Performing post-generation cleanup")
  py_gc$collect() # Trigger Python's garbage collector

  # Check the device the model is actually on
  model_device_type <- tryCatch({
    current_model$device$type
  }, error = function(e) { "unknown" }) # In case device attribute isn't standard

  if (model_device_type == "cuda") {
    cli_alert_info("Clearing CUDA cache.")
    torch$cuda$empty_cache()
  } else {
    cli_alert_info("Model is on {.val {model_device_type}}, skipping CUDA cache clear (or CUDA not used).")
  }
  cli_process_done()

  invisible()
}

# --- 3. Script Execution ---
cli_h1("Execution")

# Configuration
text_to_say <- "Suhhhhhhhhhhhhh, this is a test with improved memory management."
output_dir <- here("data", "output")
input_dir <- here("data", "input") # Assuming you have sample.wav here

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cli_alert_info("Created output directory: {.path {output_dir}}")
}
# Ensure input sample exists if you plan to use it
sample_voice_path <- here(input_dir, "sample.wav")
if (!file.exists(sample_voice_path)) {
  cli_alert_warning("Input sample voice not found at {.path {sample_voice_path}}. Custom voice generation might fail or use no prompt if path is NULL.")
  # To prevent error if chatterbox function expects a valid path or NULL:
  # sample_voice_path <- NULL
}


# Initialize Model ONCE
# Set desired_device to "cuda" to attempt GPU, or "cpu" for CPU-only.
desired_device <- "cuda"
tts_model <- NULL # Initialize to NULL

tryCatch({
  tts_model <- init_chatterbox(device_preference = desired_device)
}, error = function(e) {
  cli_abort("Fatal error during model initialization: {e$message}")
})

if (is.null(tts_model)) {
  cli_abort("Model initialization failed. Exiting.")
}

# Base voice generation
cli_h2("Base Voice Generation")
output_path_1 <- here(output_dir, "test-memory-mgm-1.wav")
tryCatch({
  chatterbox(
    text_to_speak = text_to_say,
    output_file_path = output_path_1,
    current_model = tts_model
  )
  cli_alert_success("Base voice audio generated and saved: {.path {output_path_1}}")
}, error = function(e) {
  cli_alert_danger("Error in base voice generation: {e$message}")
})


# Custom Voice generation (if sample.wav exists)
cli_h2("Custom Voice Generation")
output_path_2 <- here(output_dir, "test-memory-mgm-2.wav")

if (!is.null(sample_voice_path) && file.exists(sample_voice_path)) {
  tryCatch({
    chatterbox(
      text_to_speak = text_to_say,
      output_file_path = output_path_2,
      audio_prompt_file_path = sample_voice_path,
      current_model = tts_model
    )
    cli_alert_success("Custom voice audio generated and saved: {.path {output_path_2}}")
  }, error = function(e) {
    cli_alert_danger("Error in custom voice generation: {e$message}")
  })
} else {
  cli_alert_info("Skipping custom voice generation as sample voice file is not available.")
}

chatterbox(
  "Let's fucking party! It's the weekend bro, smoke up!",
  "test.wav",
  tts_model,
  audio_prompt_file_path = "data/input/sample.wav"
)

# --- 4. Optional Final Cleanup ---
cli_h1("Optional Final Cleanup")
if (!is.null(tts_model) && !inherits(tts_model, "python.builtin.NoneType")) {
  cli_process_start("Cleaning up TTS model from memory")
  model_device_type <- tryCatch({ tts_model$device$type }, error = function(e) { "unknown" })

  if (model_device_type == "cuda") {
    tryCatch({
      tts_model$to("cpu") # Move model to CPU first if it was on CUDA
      cli_alert_info("Model moved to CPU.")
    }, error = function(e) {
      cli_alert_warning("Could not move model to CPU: {e$message}")
    })
  }

  rm(tts_model) # Remove R's reference
  py_gc$collect() # Python GC

  if (model_device_type == "cuda" && !is.null(torch) && torch$cuda$is_available()) {
    torch$cuda$empty_cache() # Clear CUDA cache again
    cli_alert_info("Final CUDA cache clear executed.")
  }
  cli_process_done()
  cli_alert_info("Model cleanup attempted.")
} else {
  cli_alert_info("No model to clean up or model was already NULL.")
}

cli_alert_success("Script completed.")
