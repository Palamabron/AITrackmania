import platform

if platform.system() in ("Windows", "Linux"):
    import time

    import numpy as np

    def control_gamepad(gamepad, control):
        """Send a single-step driving command to a virtual gamepad.

        Sanitises the input (NaN → 0, clamp to [-1, 1]), applies a non-linear
        affine remap to the gas trigger and a sigmoid-blended tangent curve to
        steering, then pushes all three axes in one atomic ``gamepad.update()``
        call so the game sees a consistent state.

        Args:
            gamepad: A vgamepad VX360Gamepad (or compatible) instance.
            control: Sequence of three floats [gas, brake, steer].
                gas   in [0, 1] — right trigger; values ≤ 0 set the trigger to 0.
                brake in [0, 1] — left trigger; values ≤ 0.75 set the trigger to 0
                    (dead-zone to prevent unintentional braking from small noise).
                steer in [-1, 1] — left stick X-axis, passed through
                    ``mapped_steering`` before being sent to the device.
        """
        control = [float(np.clip(np.nan_to_num(c, nan=0.0), -1.0, 1.0)) for c in control]
        if control[0] > 0.0:  # gas
            mapped_value = 0.165 * control[0] + 0.835  # f(-0.515)=0.75
            gamepad.right_trigger_float(value_float=mapped_value)  # car starts driving from 0.75
        else:
            gamepad.right_trigger_float(value_float=0.0)
        if control[1] > 0.75:  # brake
            gamepad.left_trigger_float(value_float=control[1])
        else:
            gamepad.left_trigger_float(value_float=0.0)

        gamepad.left_joystick_float(mapped_steering(control[2]), 0.0)  # turn
        gamepad.update()

    def mapped_steering(x, k=15):
        """Apply a non-linear sigmoid-blend to a raw steering value.

        Near the centre the output approximates ``tan(x)/tan(1)`` (slightly
        progressive); near the edges a double-sigmoid gate suppresses the
        tangent so the output stays bounded within (-1, 1) for x in [-1, 1].

        Args:
            x: Raw steering value in [-1, 1].
            k: Sigmoid steepness (default 15). Higher values sharpen the
                transition between the progressive centre and the saturated
                edges.

        Returns:
            Mapped steering value suitable for ``left_joystick_float``,
            in (-1, 1).
        """
        sigmoid = 1 / (1 + np.exp(-k * (x + 0.4))) * 1 / (1 + np.exp(-k * (0.4 - x)))
        return np.tan(x) / np.tan(1) * (1 - sigmoid)

    def gamepad_reset(gamepad):
        """Reset the gamepad state and send a B-button press to respawn the car.

        Clears all axes, triggers, and buttons on the virtual device, then
        simulates a B-button press (code 0x2000) with a 0.1 s hold before
        releasing.  The 0.1 s hold is the minimum duration TrackMania requires
        to register a respawn input.

        Args:
            gamepad: A vgamepad VX360Gamepad (or compatible) instance.
        """
        gamepad.reset()
        gamepad.press_button(button=0x2000)  # press B button
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(button=0x2000)  # release B button
        gamepad.update()

    def gamepad_save_replay_tm20(gamepad):
        """Navigate the TM2020 replay-save menu via gamepad button presses.

        Waits 5 s for the end-of-race screen to appear, then executes the
        button sequence required by the TM2020 UI to save the current replay:
          D-pad down → A  (select "Save replay")
          D-pad up   → A  (confirm the save)

        Inter-press delays are chosen to exceed the game's UI animation
        duration between each input.

        Args:
            gamepad: A vgamepad VX360Gamepad (or compatible) instance.
        """
        time.sleep(5.0)
        gamepad.reset()
        gamepad.press_button(0x0002)  # dpad down
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x0002)  # dpad down
        gamepad.update()
        time.sleep(0.2)
        gamepad.press_button(0x1000)  # A
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x1000)  # A
        gamepad.update()
        time.sleep(0.2)
        gamepad.press_button(0x0001)  # dpad up
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x0001)  # dpad up
        gamepad.update()
        time.sleep(0.2)
        gamepad.press_button(0x1000)  # A
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x1000)  # A
        gamepad.update()

    def gamepad_close_finish_pop_up_tm20(gamepad):
        """Dismiss the TM2020 finish pop-up by pressing the A button.

        Resets all axes, presses A (code 0x1000) with a 0.1 s hold, then
        releases.  The 0.1 s hold is the minimum duration the game UI requires
        to register a dismiss input.

        Args:
            gamepad: A vgamepad VX360Gamepad (or compatible) instance.
        """
        gamepad.reset()
        gamepad.press_button(0x1000)  # A
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x1000)  # A
        gamepad.update()

else:

    def control_gamepad(gamepad, control):
        pass

    def gamepad_reset(gamepad):
        pass

    def gamepad_save_replay_tm20(gamepad):
        pass

    def gamepad_close_finish_pop_up_tm20(gamepad):
        pass
