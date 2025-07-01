%% sets the parameters of the simulation
num_of_points_in_subs_vec = 500; %number of points in each 
% direction in the simulation
max_hight_points = 600;
num_of_iterations = [85, 100, 115, 130, 145, 160, 175, 85, 100, 115, 130, 145, 160, 175]; %array of ints
min_num_seeds_per_it = 10;
max_num_seeds_per_it = 15; %in each iteration the number of randomized potential nucleation events 
% would be a random number between min_num_seeds_per_it and max_num_seeds_per_it
base_growth_steps_per_it = 2; %number of growth steps per iteration for non porous sample. 
% open porosity would lead to more growth
cone_exp = 2; %always 2 for parabolic cones
k = [2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3];
exposed_pores_return_ratio = 1; %for full ricocheting from the exposed pores choose 1. you can set 
% it to a lower number to make an incomplete ricocheting scenario
num_of_matrix_pores = [0, 0, 0, 0, 0, 0, 0, 24, 24, 24, 24, 24, 24, 24]; % array of ints
matrix_pore_rad_avg = [31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31]; %in pixels
matrix_pore_rad_std = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]; %in pixels

% make sure that the lenghts of num_of_iterations, k, num_of_matrix_pores,
% matrix_pore_rad_avg and matrix_pore_rad_std are the same

pixel_size = 0.5; %nm
num_of_iterations_per_sample_kind = 30; % int. each kind of sample will be run this amount of 
% times, with the average and stadard deviation taken
%% main loop of the simulation
subs_porousity_ave = ...
    zeros(1, length(num_of_matrix_pores));
subs_porousity_std = ...
    zeros(1, length(num_of_matrix_pores));
sputtered_porousity_ave = ...
    zeros(1, length(num_of_matrix_pores));
sputtered_porousity_std = ...
    zeros(1, length(num_of_matrix_pores));
ave_height_ave = ...
    zeros(1, length(num_of_matrix_pores));
ave_height_std = ...
    zeros(1, length(num_of_matrix_pores));
roughness_ave = ...
    zeros(1, length(num_of_matrix_pores));
roughness_std = ...
    zeros(1, length(num_of_matrix_pores));
por_at_height_ave = ...
    zeros(max_hight_points, ...
    length(num_of_matrix_pores));
por_at_height_std = ...
    zeros(max_hight_points, ...
    length(num_of_matrix_pores));
for sample_kind_it = 1:length(num_of_matrix_pores)
    sample_subs_porousity = ...
        zeros(1, num_of_iterations_per_sample_kind);
    sample_sput_porousity = ...
        zeros(1, num_of_iterations_per_sample_kind);
    ave_height = ...
        zeros(1, num_of_iterations_per_sample_kind);
    roughness = ...
        zeros(1, num_of_iterations_per_sample_kind);
    por_per_height_inst = ...
        zeros(max_hight_points, num_of_iterations_per_sample_kind);
    for sample = 1:num_of_iterations_per_sample_kind
        [surface_porosity, sputtered_porosity, ...
            avg_height, std_height, por_per_height] = ...
            porous_sputtering_one_time_plus_until_height_fall_growth_var...
            (num_of_points_in_subs_vec, max_hight_points, ...
            min_num_seeds_per_it, max_num_seeds_per_it, ...
            base_growth_steps_per_it, cone_exp, ...
            num_of_matrix_pores(sample_kind_it), ...
            matrix_pore_rad_avg(sample_kind_it), ...
            matrix_pore_rad_std(sample_kind_it), k(sample_kind_it), ...
            exposed_pores_return_ratio, ...
            num_of_iterations(sample_kind_it));
        sample_subs_porousity(sample) = surface_porosity;
        sample_sput_porousity(sample) = sputtered_porosity;
        ave_height(sample) = avg_height;
        for height_point = 1:max_hight_points
            por_per_height_inst(height_point, sample) = ...
                por_per_height(height_point);
        end
        roughness(sample) = std_height;
        disp(sample)
    end
    subs_porousity_ave(sample_kind_it) = ...
        mean(sample_subs_porousity);
    subs_porousity_std(sample_kind_it) = ...
        std(sample_subs_porousity);
    sputtered_porousity_ave(sample_kind_it) = ...
        mean(sample_sput_porousity);
    sputtered_porousity_std(sample_kind_it) = ...
        std(sample_sput_porousity);
    ave_height_ave(sample_kind_it) = ...
        mean(ave_height);
    ave_height_std(sample_kind_it) = ...
        std(ave_height);
    roughness_ave(sample_kind_it) = ...
        mean(roughness);
    roughness_std(sample_kind_it) = ...
        std(roughness);
    for height_point = 1:max_hight_points
        porosity_in_this_height_and_sample_kind = ...
            zeros(1, num_of_iterations_per_sample_kind);
        for sample = 1:num_of_iterations_per_sample_kind
            porosity_in_this_height_and_sample_kind(sample)...
                = por_per_height_inst(height_point, sample);
        end
        por_at_height_ave(height_point, sample_kind_it) ...
            = mean(porosity_in_this_height_and_sample_kind);
        por_at_height_std(height_point, sample_kind_it) ...
            = std(porosity_in_this_height_and_sample_kind);
    end
    disp(sample_kind_it)
end
subs_porousity_percent_ave = subs_porousity_ave * 100
subs_porousity_percent_std = subs_porousity_std * 100
sputtered_porousity_percent_ave = ...
    sputtered_porousity_ave * 100
sputtered_porousity_percent_std = ...
    sputtered_porousity_std * 100

%% plots the graphs
% sputtered porosity v. substrate porososity
figure(1)
plot(subs_porousity_percent_ave, ...
    sputtered_porousity_percent_ave, ".")
xlabel('Substrate porousity [%]')
ylabel('Sputtered porousity [%]')
hold on
errorbar(subs_porousity_percent_ave, ...
    sputtered_porousity_percent_ave, ...
    sputtered_porousity_percent_std, ...
    "LineStyle","none")
hold on
errorbar(subs_porousity_percent_ave, ...
    sputtered_porousity_percent_ave, ...
    subs_porousity_percent_std, ...
    "horizontal","LineStyle","none")
hold off
% average height v. substrate porososity
figure(2)
ave_height_ave_stnd = ave_height_ave*pixel_size;
ave_height_std_stnd = ave_height_std*pixel_size;
plot(subs_porousity_percent_ave, ave_height_ave_stnd, ".")
xlabel('Substrate porousity [%]')
ylabel('Average height [nm]')
hold on
errorbar(subs_porousity_percent_ave, ...
    ave_height_ave_stnd, ...
    subs_porousity_percent_std, ...
    "horizontal", "LineStyle","none")
hold on
errorbar(subs_porousity_percent_ave, ...
    ave_height_ave_stnd, ...
    ave_height_std_stnd, ...
    "LineStyle","none")
hold off
% roughness v. substrate porososity
figure(3)
roughness_ave_stnd = roughness_ave*pixel_size;
roughness_std_stnd = roughness_std*pixel_size;
plot(subs_porousity_percent_ave, roughness_ave_stnd, ".")
xlabel('Substrate porousity [%]')
ylabel('Roughness [nm]')
hold on
errorbar(subs_porousity_percent_ave, ...
    roughness_ave_stnd, ...
    subs_porousity_percent_std, ...
    "horizontal", "LineStyle","none")
hold on
errorbar(subs_porousity_percent_ave, ...
    roughness_ave_stnd, ...
    roughness_std_stnd, ...
    "LineStyle","none")
hold off
% sputtered porosity v. average pore size
figure(4)
matrix_pore_rad_avg_nm = matrix_pore_rad_avg.*pixel_size;
plot(matrix_pore_rad_avg_nm, ...
    sputtered_porousity_percent_ave, ".")
xlabel('Average pore size [nm]')
ylabel('Sputtered porousity [%]')
hold on
errorbar(matrix_pore_rad_avg_nm, ...
    sputtered_porousity_percent_ave, ...
    sputtered_porousity_percent_std, ...
    "LineStyle","none")
hold off
% average height v. average pore size
figure(5)
plot(matrix_pore_rad_avg_nm, ave_height_ave_stnd, ".")
xlabel('Average pore size [nm]')
ylabel('Average height [nm]')
hold on
errorbar(matrix_pore_rad_avg_nm, ...
    ave_height_ave_stnd, ...
    ave_height_std_stnd, ...
    "LineStyle","none")
hold off
% roughness v. average pore size
figure(6)
plot(matrix_pore_rad_avg_nm, roughness_ave_stnd, ".")
xlabel('Average pore size [nm]')
ylabel('Roughness [nm]')
hold on
errorbar(matrix_pore_rad_avg_nm, ...
    roughness_ave_stnd, ...
    roughness_std_stnd, ...
    "LineStyle","none")
hold off
% sputtered porosity v. number of iterations
figure(7)
plot(num_of_iterations, ...
    sputtered_porousity_percent_ave, ".")
xlabel('number of iterations')
ylabel('Sputtered porousity [%]')
hold on
errorbar(num_of_iterations, ...
    sputtered_porousity_percent_ave, ...
    sputtered_porousity_percent_std, ...
    "LineStyle","none")
hold off
% average height v. number of iterations
figure(8)
plot(num_of_iterations, ave_height_ave_stnd, ".")
xlabel('number of iterations')
ylabel('Average height [nm]')
hold on
errorbar(num_of_iterations, ...
    ave_height_ave_stnd, ...
    ave_height_std_stnd, ...
    "LineStyle","none")
hold off
% roughness v. number of iterations
figure(9)
plot(num_of_iterations, roughness_ave_stnd, ".")
xlabel('number of iterations')
ylabel('Roughness [nm]')
hold on
errorbar(num_of_iterations, ...
    roughness_ave_stnd, ...
    roughness_std_stnd, ...
    "LineStyle","none")
hold off
% sputtered porosity v. standard divaition of sputtered pore size
figure(10)
matrix_pore_rad_std_nm = matrix_pore_rad_std.*pixel_size;
plot(matrix_pore_rad_std_nm, ...
    sputtered_porousity_percent_ave, ".")
xlabel('Pore size standard deviation [nm]')
ylabel('Sputtered porousity [%]')
hold on
errorbar(matrix_pore_rad_std_nm, ...
    sputtered_porousity_percent_ave, ...
    sputtered_porousity_percent_std, ...
    "LineStyle","none")
hold off
% average height v. standard divaition of sputtered pore size
figure(11)
plot(matrix_pore_rad_std_nm, ave_height_ave_stnd, ".")
xlabel('Pore size standard deviation [nm]')
ylabel('Average height [nm]')
hold on
errorbar(matrix_pore_rad_std_nm, ...
    ave_height_ave_stnd, ...
    ave_height_std_stnd, ...
    "LineStyle","none")
hold off
% roughness v. standard divaition of sputtered pore size
figure(12)
plot(matrix_pore_rad_std_nm, roughness_ave_stnd, ".")
xlabel('Pore size standard deviation [nm]')
ylabel('Roughness [nm]')
hold on
errorbar(matrix_pore_rad_std_nm, ...
    roughness_ave_stnd, ...
    roughness_std_stnd, ...
    "LineStyle","none")
hold off
% 2D porosity v. height
figure(13)
height_point_nums_vec = linspace(1, max_hight_points, max_hight_points);
heights_vec = height_point_nums_vec * pixel_size;
% Created manually. The point where the porosity at height starts to climb
% due to grains not reching this height
cutoffs_vec = [360, 360, 360, 360, 360, 360];
legend_entry_1 = "Surface porosity = " + ...
    int2str(subs_porousity_percent_ave(1)) + "%";
legend_entry_2 = "Surface porosity = " + ...
    int2str(subs_porousity_percent_ave(2)) + "%";
legend_entry_3 = "Surface porosity = " + ...
    int2str(subs_porousity_percent_ave(3)) + "%";
legend_entry_4 = "Surface porosity = " + ...
    int2str(subs_porousity_percent_ave(4)) + "%";
legend_entry_5 = "Surface porosity = " + ...
    int2str(subs_porousity_percent_ave(5)) + "%";
legend_entry_6 = "Surface porosity = " + ...
    int2str(subs_porousity_percent_ave(6)) + "%";
plot(heights_vec(1:cutoffs_vec(1)), ...
    por_at_height_ave(1:cutoffs_vec(1), 1),...
    heights_vec(1:cutoffs_vec(2)), ...
    por_at_height_ave(1:cutoffs_vec(2), 2),...
    heights_vec(1:cutoffs_vec(3)), ...
    por_at_height_ave(1:cutoffs_vec(3), 3),...
    heights_vec(1:cutoffs_vec(4)), ...
    por_at_height_ave(1:cutoffs_vec(4), 4),...
    heights_vec(1:cutoffs_vec(5)), ...
    por_at_height_ave(1:cutoffs_vec(5), 5),...
    heights_vec(1:cutoffs_vec(6)), ...
    por_at_height_ave(1:cutoffs_vec(6), 6))
xlabel('height [nm]')
ylabel('porosity at height [%]')
legend(legend_entry_1, legend_entry_2, ...
    legend_entry_3, legend_entry_4, ...
    legend_entry_5, legend_entry_6, ...
    'Location','northeast')

%% functions
%% main function in the simulation, perfomes one run of it on a sample
function [surface_porosity, sputtered_porosity, avg_height, std_height, por_per_height] = ...
    porous_sputtering_one_time_plus_until_height_fall_growth_var...
    (num_of_points_in_subs_vec, max_hight_points, min_num_seeds_per_it,...
    max_num_seeds_per_it, base_growth_steps_per_it, cone_exp, num_of_matrix_pores, matrix_pore_rad_avg, ...
    matrix_pore_rad_std, k, exposed_pores_return_ratio, num_of_iterations)
growth_matrix = zeros(num_of_points_in_subs_vec, ...
    num_of_points_in_subs_vec, max_hight_points); %0 
% for no growth in this place (at least not yet), 1 
% for growth
allowed_places_for_seed = ones(num_of_points_in_subs_vec, ...
    num_of_points_in_subs_vec); %1 for allowed 0 for not
active_seeds = [];

next_it_growth = zeros(num_of_points_in_subs_vec, ...
    num_of_points_in_subs_vec, 3);
%{
this is a matrix that dictates in which places the next 
growth steps would cause:
rows and colomns - the row and colomn
next_it_growth(i,j,1) - 
1 for growth in the i,j place, 0 for no growth
next_it_growth(i,j,2) - the height of the growth if 
next_it_growth(i,j,1) = 1
next_it_growth(i,j,3) - the grain that caused the 
growth if(i,j,1) = 1
%}


% randomly puts pores in the matrix
if num_of_matrix_pores > 0
    pore_centers = zeros(2, num_of_matrix_pores);
end
for pore_num = 1:num_of_matrix_pores
    x_pore = randi(num_of_points_in_subs_vec);
    y_pore = randi(num_of_points_in_subs_vec);
    matrix_pore_rad = max(round(normrnd(matrix_pore_rad_avg, matrix_pore_rad_std)), 0);
    if num_of_matrix_pores > 0
        pore_centers(1, pore_num) = x_pore;
        pore_centers(2, pore_num) = y_pore;
    end
    for i = max(x_pore-matrix_pore_rad,1):...
            min(x_pore+matrix_pore_rad,num_of_points_in_subs_vec)
        for j = max(y_pore-matrix_pore_rad,1):...
                min(y_pore+matrix_pore_rad,num_of_points_in_subs_vec)
            if sqrt((x_pore-i)^2+(y_pore-j)^2) <= matrix_pore_rad
                allowed_places_for_seed(i,j) = 0;
            end
        end
    end
end
init_pore_mapping = allowed_places_for_seed;
exposed_pores = zeros(num_of_points_in_subs_vec, ...
    num_of_points_in_subs_vec);
for i = 1:num_of_points_in_subs_vec
    for j = 1:num_of_points_in_subs_vec
        if allowed_places_for_seed(i, j) == 0
            exposed_pores(i, j) = 1;
        end
    end
end
%calculates matrix porosity
num_of_init_available_spaces = 0;
for i = 1:num_of_points_in_subs_vec
    for j = 1:num_of_points_in_subs_vec
        num_of_init_available_spaces = num_of_init_available_spaces + ...
            allowed_places_for_seed(i,j);
    end
end
potential_seed_places_num = num_of_points_in_subs_vec*num_of_points_in_subs_vec;
surface_porosity = 1 - num_of_init_available_spaces/...
    potential_seed_places_num;
%disp(surface_porosity);
%draw_height_matrix(growth_matrix, ...
%    init_pore_mapping, max_hight_points, 1);
total_growth_steps = 0;
%it = 0;
for it = 1:num_of_iterations
    %it = it + 1;
    %new seeds
    it_seed_num = randi([min_num_seeds_per_it, ...
        max_num_seeds_per_it]);
    for new_seed_num = 1:it_seed_num
        i_new_seed = randi(num_of_points_in_subs_vec);
        j_new_seed = randi(num_of_points_in_subs_vec);
        if allowed_places_for_seed(i_new_seed, j_new_seed) == 1 %it is an allowed seed
            max_width = inf;
            active_seeds = [active_seeds; i_new_seed j_new_seed 0 max_width];
            next_it_growth(i_new_seed, j_new_seed, 1) = 1;
            next_it_growth(i_new_seed, j_new_seed, 2) = 1;
            next_it_growth(i_new_seed, j_new_seed, 3) = ...
                length(active_seeds(:, 1));
        end
    end
    num_of_exposed_surface_pore_places = 0;
    for i = 1:num_of_points_in_subs_vec
        for j = 1:num_of_points_in_subs_vec
            if exposed_pores(i, j) == 1
                num_of_exposed_surface_pore_places = ...
                    num_of_exposed_surface_pore_places + 1;
            end
        end
    end
    exposed_surface_pores_percent = ...
        num_of_exposed_surface_pore_places / ...
        num_of_points_in_subs_vec^2;
    unrounded_growth_steps_per_it = ...
        base_growth_steps_per_it * ...
        (1 + exposed_surface_pores_percent * ...
        exposed_pores_return_ratio);
    growth_steps_per_it = ...
        weight_random_round...
        (unrounded_growth_steps_per_it);
    total_growth_steps = total_growth_steps + ...
        growth_steps_per_it;
    %grain growth
    for grain_growth_step = 1:growth_steps_per_it
        next_it_growth_it_start = next_it_growth;
        for i = 1:num_of_points_in_subs_vec
            for j = 1:num_of_points_in_subs_vec
                if next_it_growth_it_start(i,j,1) == 1
                    growth_matrix(i,j, next_it_growth_it_start(i,j,2)) = 1;
                    allowed_places_for_seed(i,j) = 0;
                    next_it_growth = ...
                        change_next_growth_new...
                        (next_it_growth, ...
                        next_it_growth_it_start, ...
                        active_seeds, i, j, ...
                        num_of_points_in_subs_vec, k);
                    exposed_pores(i, j) = 0;
                end
            end
        end
    end
    %draws the height profile every 10 iterations
    
    if mod(it, 10) == 0
        disp('!')
    end
    
end
%caluculates porosity and average height
height_matrix = create_height_matrix(growth_matrix, ...
    init_pore_mapping, max_hight_points);
tot_num_of_spaces = 0;
num_of_filled_spaces = 0;
num_of_empty_xy_places = 0;
height_points = [];
for i = 1:num_of_points_in_subs_vec
    for j = 1:num_of_points_in_subs_vec
        if height_matrix(i, j) > 0
            height_points = [height_points, height_matrix(i, j)];
            tot_num_of_spaces = tot_num_of_spaces + height_matrix(i, j);
        else
            num_of_empty_xy_places = num_of_empty_xy_places + 1;
        end
        for k = 1:max(height_matrix(i, j), 1)
            num_of_filled_spaces = num_of_filled_spaces + ...
                growth_matrix(i,j,k);
        end
    end
end
avg_height = mean(height_points);
std_height = std(height_points);
sputtered_porosity = 1 - num_of_filled_spaces/(avg_height * num_of_points_in_subs_vec^2);
%calculate porosity per height
por_per_height = zeros(1, max_hight_points);
for height = 1:max_hight_points
    num_filled_spaces_in_hieght = 0;
    num_of_points_in_plane = num_of_points_in_subs_vec*num_of_points_in_subs_vec;
    for i = 1:num_of_points_in_subs_vec
        for j = 1:num_of_points_in_subs_vec
            num_filled_spaces_in_hieght = num_filled_spaces_in_hieght + ...
                growth_matrix(i, j, height);
        end
    end
    por_at_height_percent = (1 - num_filled_spaces_in_hieght / num_of_points_in_plane) * 100;
    por_per_height(height) = por_at_height_percent;
end
end

%% rounds a nuber to the interger above or below it,
% where the probability to get the interger above it is how close it is to
% it. for instance, if the number is 5.9 it will round to 6 90% of the
% times and to 5 the remaining 10%
function roundedNumber = weight_random_round(number)
% Determine the integer part and fractional part of the number
integerPart = floor(number);
fractionalPart = number - integerPart;
    
% Define the probabilities for rounding down and rounding up
% Adjust these probabilities as needed
probRoundDown = 1 - fractionalPart; % probability of rounding down
    
% Generate a random number to decide rounding direction
if rand() < probRoundDown
    % Round down
    roundedNumber = integerPart;
else
    % Round up
    roundedNumber = ceil(number);
end
end

%% changes the next_it_growth matrix for the next growth step
function next_it_growth = change_next_growth_new(next_it_growth, next_it_growth_it_start, active_seeds, i,...
    j, num_of_points_in_subs_vec, k)
h = next_it_growth_it_start(i, j, 2) + 1;
roundup_k = ceil(k);
grain_num = next_it_growth_it_start(i, j, 3);
for di = -roundup_k:roundup_k
    for dj = -roundup_k:roundup_k
        % Check if the current element is within the matrix bounds
        if (i + di >= 1 && i + di <= num_of_points_in_subs_vec) && (j + dj >= 1 && j + dj <= num_of_points_in_subs_vec)
            % Apply the function to the matrix element (i+di, j+dj)
            next_it_growth = ...
                change_next_growth_one_point(next_it_growth, active_seeds, i + di, j + dj, h, k, grain_num);
        end
    end
end
end

%% changes the next_it_growth as a result of one grain
function next_it_growth = change_next_growth_one_point(next_it_growth,...
    active_seeds, i, j, h, k, grain_num)
if next_it_growth(i, j, 2) < h
    seed_i = active_seeds(grain_num, 1);
    seed_j = active_seeds(grain_num, 2);
    radius_at_height = k * sqrt(h);  % Compute radius at height h
        
    % Calculate distance using vectorized operations
    distance = sqrt((seed_i - i)^2 + (seed_j - j)^2);
        
    % Check if distance is within the radius at height h
    if distance < radius_at_height
        % Update next_it_growth only if the condition is met
        next_it_growth(i, j, :) = [1, h, grain_num];
    end
end
end

%% creates a mtrix of the height ot the sputterred layer in each point
function height_matrix = create_height_matrix(growth_matrix, init_pore_mapping, max_hight_points)
mat_size = size(growth_matrix);
height_matrix = zeros(mat_size(1), mat_size(2));
for i = 1:mat_size(1)
    for j = 1:mat_size(2)
        height_of_point = max_hight_points;
        while height_of_point > 0
            if growth_matrix(i, j, height_of_point) == 1
                break
            else
                height_of_point = height_of_point - 1;
            end
        end
        if height_of_point > 0
            height_matrix(i,j) = height_of_point;
        else
            if init_pore_mapping(i,j) == 1
                height_matrix(i,j) = 0;
            else
                height_matrix(i,j) = -1;
            end
        end
    end
end
end