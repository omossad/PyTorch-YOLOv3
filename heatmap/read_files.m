function [x,y] = read_files(game_name)
%Read CSV files and return the x,y coordinates of the fixation
%   This function reads the name of the game and opens all csv
%   files containing the eye fixations, next it return the x,y
%   coordinates of the eye fixation
x = [];
y = [];
game_directory = strcat('fixations/', game_name);

files = dir(game_directory);
for f=1:length(files)
    if files(f).isdir == 0
        filename = files(f).name;
        file_directory = strcat(game_directory, '/');
        file_directory = strcat(file_directory, filename);
        data = readtable(file_directory);
        %data = data{:,:};
        x = [x; data{:,3}];
        y = [y; data{:,4}];
    end
end
 
end

