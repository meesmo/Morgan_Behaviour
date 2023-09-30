import time
import RPi.GPIO as GPIO
import numpy as np
import threading
import core
from picamera import PiCamera
from pygame import mixer

protocol_name = '20Hz_opto_classical_single'
protocol_description = ('Mice lick single port after hearing the tone, opto is shone during 2s trace period, aim to see lick activity change during opto stimulation in 5-HT knockouts')

camera = PiCamera()  # Create camera object
camera.start_preview(fullscreen=False, window=(0, -44, 350, 400))

# ------------------------------------------------------------------------------
# Set experimental parameters:
# ------------------------------------------------------------------------------

experimenter = input('Initials: ')
mouse_number = input('mouse number: ')
mouse_weight = float(input('mouse weight(g): '))


block_number = input('block number: ')
n_trials = int(input('How many trials?: '))
ttl_experiment = input('Send trigger pulses to opto-stim? (y/n): ')
syringe_check = input('Syringe check: ')

response_window = 4000 # total time of one entire trial

sample_tone_length = 1  # Length of sample tone (s)
tone_freq = 3000  # Frequency(Hz) of single sample tone.

end_tone_freq = 2500
end_tone_length = 8

reward_size = 10  # Volume(uL) of water rewards.

opto_freq = 20

# ------------------------------------------------------------------------------
# Assign GPIO pins:
# ------------------------------------------------------------------------------

servo_PWM = 17  # PWM pin for servo that adjusts lickport distance

L_enablePIN = 23  # Enable pin for left stepper motor
L_directionPIN = 24  # Direction pin for left stepper motor
L_stepPIN = 25  # Step pin for left stepper motor
L_emptyPIN = 20  # Empty switch pin for left stepper motor
L_lickometer = 12  # Input pin for lickometer (black wire)

R_enablePIN = 10  # Enable pin for right stepper motor
R_directionPIN = 9  # Direction pin for right stepper motor
R_stepPIN = 11  # Step pin for right stepper motor
R_emptyPIN = 21  # Empty switch pin for right stepper motor
R_lickometer = 16  # Input pin for lickometer (black wire)

TTL_opto_PIN = 15  # output for TTL pulse for opto stim

# ------------------------------------------------------------------------------
# Initialize class instances for experiment:
# ------------------------------------------------------------------------------

# Turn off the GPIO warnings
GPIO.setwarnings(False)

# Set the mode of the pins (broadcom vs local)
GPIO.setmode(GPIO.BCM)

# Set the enable pins for L and R stepper motors to 1 to prevent overheating
GPIO.setup(L_enablePIN, GPIO.OUT, initial=1)
GPIO.setup(R_enablePIN, GPIO.OUT, initial=1)

# Initialize the mixer (for tones) at the proper sampling rate.
mixer.init(frequency=44100)

# Create Stepper class instances for right reward delivery
water_R = core.stepper(R_enablePIN, R_directionPIN, R_stepPIN, R_emptyPIN)

# Create lickometer class instances for right lickometers
lick_port_R = core.lickometer(R_lickometer)

# Create instruction tones
tonefreq = core.PureTone(tone_freq, sample_tone_length)

# Create tone that is used as an error signal
#tone_wrong = core.PureTone(wrong_tone_freq, wrong_tone_length)
#tone_end = core.PureTone(end_tone_freq, end_tone_length, vol=-25)


if ttl_experiment == 'y':
    # Set up ttl class instances opto stim TTL output
    TTL_opto = core.ttl(TTL_opto_PIN, opto_freq, ISI_length = 0.045)

# ------------------------------------------------------------------------------
# Initialize experiment:
# ------------------------------------------------------------------------------

# Set the time for the beginning of the block
trials = np.arange(n_trials)
data = core.data(protocol_name, protocol_description, n_trials, mouse_number,
                 block_number, experimenter, mouse_weight)
data.opto_trial = np.empty(len(trials), dtype=np.bool)

total_reward_R = 0
supp_reward_R = 0
performance = 0  # Total number of correct responses (to print at each trial)

# -------------------------------------------------------------------------------
# Iterate through trials:
# -------------------------------------------------------------------------------

# Start imaging laser scanning

for trial in trials:
    # initialize all the things
    data._t_start_abs[trial] = time.time() * 1000  # Time at beginning of trial
    data.t_start[trial] = data._t_start_abs[trial] - data._t_start_abs[0]
    data.sample_tone[trial] = 'R'
    data.t_sample_tone[trial] = time.time() * 1000 - data._t_start_abs[trial]

    tone = rule.R_tone
    if trial < 30:
        _opto_trial = False
    elif trial > 30:
        _opto_trial = np.random.rand() < 0.5

    data.opto_trial[trial] = _opto_trial

    thread_R = threading.Thread(target=lick_port_R.Lick, args=(1000, 8))
    thread_ttl = threading.Thread(target=TTL_opto.deliver_pulses)
    thread_R.start()

    # wait for baseline
    time.sleep(2)

    # play tone
    tone.play()  # Play left tone
    data.sample_tone_end[trial] = (time.time() * 1000
                                - data._t_start_abs[trial])

    # post-tone response period: opto stim if necessary
    if _opto_trial is True:
        thread_ttl.start()
    elif _opto_trial is False:
        time.sleep(2)

    # Stochastic reward delivery
    water_R.Reward()

    data.t_rew_r[trial] = (time.time() * 1000
                        - data._t_start_abs[trial])
    data.v_rew_r[trial] = reward_size

    # log end time
    data.t_end[trial] = time.time() * 1000 - data._t_start_abs[0]

    # -------------------------------------------------------------------------
    # Post-trial data storage
    # -------------------------------------------------------------------------
    # Make sure the threads are finished
    thread_R.join()
    thread_ttl.join()

    lick_port_R._t_licks -= data._t_start_abs[trial]

    # Store and process the data
    storage_list = [data.lick_r]
    rawdata_list = [lick_port_R]

    for ind, storage in enumerate(storage_list):
        storage[trial] = {}
        storage[trial]['t'] = rawdata_list[ind]._t_licks
        storage[trial]['volt'] = rawdata_list[ind]._licks

    data.freq[trial] = tone.freq  # Store tone frequency.
    data.loc[trial] = tone.loc  # Store multipulse(1) or single pulse(0).

    # If freq rule, left_port=1 means highfreq on left port
    # If pulse rule, left_port=1 means multipulse on left port

    licks_detected = ''
    # Will indicate which ports recorded any licks in the entire trial.
    if sum(lick_port_R._licks) != 0:
        licks_detected += 'R'

    print(f'Tone:{tone.freq}, Resp:{response}, Licks:{licks_detected}, '
          f'Rew:{np.nansum([data.v_rew_l[trial], data.v_rew_r[trial]])}')

    # -------------------------------------------------------------------------
    # Deliver supplementary rewards:
    # -------------------------------------------------------------------------

    ITI_ = 0
    while ITI_ > 30 or ITI_ < 10:
        ITI_ = np.random.exponential(scale=20)

    data.iti_length[trial] = ITI_
    time.sleep(ITI_)

tone_end.play()
camera.stop_preview()

total_reward_L = np.nansum(data.v_rew_l)
supp_reward_L = np.nansum(data.v_rew_l_supp)
total_reward_R = np.nansum(data.v_rew_r)
supp_reward_R = np.nansum(data.v_rew_r_supp)
print(f'Total L reward: {total_reward_L} uL + {supp_reward_L}')
print(f'Total R reward: {total_reward_R} uL + {supp_reward_R}')
data.total_reward = (total_reward_L + supp_reward_L
                     + total_reward_R + supp_reward_R)
print(f'Total reward: {data.total_reward}uL')

# Ask the user if there were any problems with the experiment. If so, prompt
# the user for an explanation that will be stored in the data file.
data.exp_quality = input('Should this data be used? (y/n): ')
if data.exp_quality == 'n':
    data.exp_msg = input('What went wrong?: ')

# Store the data in an HDF5 file and upload this file to a remote drive.
data.Store()
data.Rclone()

# Delete the .wav files created for the experiment
core.delete_tones()
