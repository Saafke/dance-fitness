"""
Script to create the qualitative comparison video.
"""
#import ffmpeg
import moviepy
from moviepy.editor import VideoFileClip, clips_array, vfx, TextClip, CompositeVideoClip

def get_paths_from_squat_index(index):

    # input
    input_vid = '/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/QMUL_DANCE_DATA/squats/50-clips/RESULTS/decomr/images/{}/output.mp4'.format(index)

    # decomr
    decomr_vid = '/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/QMUL_DANCE_DATA/squats/50-clips/render-results/decomr/{}/output.mp4'.format(index)

    # expose
    expose_vid = '/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/QMUL_DANCE_DATA/squats/50-clips/render-results/expose/{}/output.mp4'.format(index)

    # romp
    romp_vid = '/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/QMUL_DANCE_DATA/squats/50-clips/render-results/romp/{}/output.mp4'.format(index)

    # vibe
    vibe_vid = '/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/QMUL_DANCE_DATA/squats/50-clips/render-results/vibe/{}/output.mp4'.format(index)

    # videopose3d
    videopose3d_vid = '/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/QMUL_DANCE_DATA/squats/50-clips/RESULTS/videopose3d/videos/{}/final_output.mp4'.format(index)

    return [input_vid, decomr_vid, expose_vid, romp_vid, vibe_vid, videopose3d_vid]

def get_clips_from_paths(paths, margin=0):
    clips = []
    for path in paths:
        if margin == 0:
            clips.append(VideoFileClip(path))
        else:
            clips.append(VideoFileClip(path).margin(margin))
    return clips

def get_max_duration(clips):
    durations = []
    for clip in clips:
        durations.append(clip.duration)
    
    print("Durations:", durations)

    # get the max duration (i.e. longest clip)
    max_duration = max(durations)
    
    #NOTE: gonna take the second video, 
    # because the input video is somehow always longer (even though nr of frames are same as the rendered stuff)
    #max_duration = durations[1]

    return max_duration

def make_clips_equal_length(clips, max_duration):

    # Slow down each clip to match longest clip
    new_clips = []
    for clip in clips:
        new_clips.append(clip.fx( vfx.speedx, clip.duration/max_duration))

    return new_clips

def add_text(clips, colors, texts, fontsize=75):
    
    result = []
    for idx, clip in enumerate(clips):
        # Generate a text clip 
        print(texts[idx])
        print(colors[idx])
        
        txt_clip = TextClip(texts[idx], fontsize = fontsize, color = colors[idx]) 
        
        print("here")
        
        # setting position of text in the center and duration will be 10 seconds 
        #txt_clip = txt_clip.set_pos(('left', 'top')).set_duration(clip.duration) 
        txt_clip = txt_clip.set_position((0.05,0.05), relative=True).set_duration(clip.duration) 
            
        # Overlay the text clip on the first video clip 
        video = CompositeVideoClip([clip, txt_clip]) 

        result.append(video)

    return result

for index in range(00,50):
    index = "{:02d}".format(index)
    
    # -- Get the paths to the renders
    # student
    front_paths = get_paths_from_squat_index(index)

    # -- Convert to Moviepy
    front_clips = get_clips_from_paths(front_paths, margin=10)

    # -- Get max durations
    front_max_duration = get_max_duration(front_clips)
    max_duration = front_max_duration
    print("final max duration", max_duration)

    # -- Slow down each video clip to match the longest clip
    front_clips = make_clips_equal_length(front_clips, max_duration)

    # Add text
    front_clips = add_text(front_clips, colors=["white", "black", "black", "black", "black", "black"],
                                texts= ["Input", "A", "B", "E", "D", "C"])
    final_clip = clips_array([
                            [front_clips[0], front_clips[1], front_clips[2]], 
                            [front_clips[5], front_clips[4], front_clips[3]],
                            ])

    # -- Save the result
    final_clip.write_videofile("./qual-results/{}.mp4".format(index), audio=False)


# # -- Calculate durations
# durations = []
# print(clip1.duration); durations.append(clip1.duration)
# print(clip2.duration); durations.append(clip2.duration)
# print(clip3.duration); durations.append(clip3.duration)
# print(clip4.duration); durations.append(clip4.duration)
# print(clip5.duration); durations.append(clip5.duration)
# print(clip6.duration); durations.append(clip6.duration)
# max_duration = max(durations)
# print("Max duration is", max_duration)

# print("\n")
# # -- Speed up each video clip to match the longest clip
# final1 = clip1.fx( vfx.speedx, clip1.duration/max_duration); print(clip1.duration)
# final2 = clip2.fx( vfx.speedx, clip2.duration/max_duration); print(clip2.duration)
# final3 = clip3.fx( vfx.speedx, clip3.duration/max_duration); print(clip3.duration)
# final4 = clip4.fx( vfx.speedx, clip4.duration/max_duration); print(clip4.duration)
# final5 = clip5.fx( vfx.speedx, clip5.duration/max_duration); print(clip5.duration)
# final6 = clip6.fx( vfx.speedx, clip6.duration/max_duration); print(clip6.duration)

# print("\n")
# # -- Calculate durations
# print(final1.duration)
# print(final2.duration)
# print(final3.duration)
# print(final4.duration)
# print(final5.duration)
# print(final6.duration)

#final_clip = clips_array([[final1, final2, final3, final4, final5, final6]])
#final_clip.resize(width=480).write_videofile("my_stack.mp4")