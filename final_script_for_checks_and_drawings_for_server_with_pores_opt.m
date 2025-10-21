diary('final_script_log.txt');
try
num_of_points_in_subs_vec = 1000; %number of points in each 
% direction in the simulation
max_hight_points = 400;
min_num_seeds_per_it = 10;
max_num_seeds_per_it = 15;
base_growth_steps_per_it = 4;
cone_exp = 2;
exposed_pores_return_ratio = 1;
k = 3.3;
num_of_matrix_pores = 24;
matrix_pore_rad_ave = 62;
matrix_pore_rad_std = 12;
growth_matrix = false(num_of_points_in_subs_vec, ...
    num_of_points_in_subs_vec, max_hight_points); %0 
% for no growth in this place (at least not yet), 1 
% for growth
allowed_places_for_seed = ones(num_of_points_in_subs_vec, ...
    num_of_points_in_subs_vec); %1 for allowed 0 for not
%min_num_seeds_per_it = 8;
active_seeds = [];

next_it_growth = zeros(num_of_points_in_subs_vec, ...
    num_of_points_in_subs_vec, 3);
% randomly puts pores in the matrix
if num_of_matrix_pores > 0
    pore_centers = zeros(2, num_of_matrix_pores);
end
for pore_num = 1:num_of_matrix_pores
    x_pore = randi(num_of_points_in_subs_vec);
    y_pore = randi(num_of_points_in_subs_vec);
    matrix_pore_rad = round(normrnd(matrix_pore_rad_ave, ...
        matrix_pore_rad_std));
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
%{
for i = 1:num_of_points_in_subs_vec
    for j = 1:num_of_points_in_subs_vec
        if allowed_places_for_seed(i, j) == 0
            exposed_pores(i, j) = 1;
        end
    end
end
%}
exposed_pores = allowed_places_for_seed == 0;
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
draw_height_matrix(growth_matrix, ...
    init_pore_mapping, max_hight_points, 1);

total_growth_steps = 0;
it = 0;
while total_growth_steps < max_hight_points
    it = it + 1;
    %new seeds
    it_seed_num = randi([min_num_seeds_per_it, ...
        max_num_seeds_per_it]);
    for new_seed_num = 1:it_seed_num
        i_new_seed = randi(num_of_points_in_subs_vec);
        j_new_seed = randi(num_of_points_in_subs_vec);
        if allowed_places_for_seed(i_new_seed, j_new_seed) == 1 %it is an allowed seed
            % calculates the distance to the nearest edge
            %if num_of_matrix_pores > 0
            %    distances_to_pore_centers = zeros(1,num_of_matrix_pores);
            %    for pore_dist_num = 1:num_of_matrix_pores
            %        distances_to_pore_centers(pore_dist_num) = ...
            %            sqrt((i_new_seed - pore_centers(1, pore_dist_num))^2 + ...
            %            (j_new_seed - pore_centers(2, pore_dist_num))^2);
            %    end
            %    max_width = min(distances_to_pore_centers) - matrix_pore_rad;
            %else
            %    max_width = inf;
            %end
            %max_width = inf;
            active_seeds = [active_seeds; i_new_seed j_new_seed];
            next_it_growth(i_new_seed, j_new_seed, 1) = 1;
            next_it_growth(i_new_seed, j_new_seed, 2) = 1;
            next_it_growth(i_new_seed, j_new_seed, 3) = ...
                length(active_seeds);
        end
    end
    %calculaltes the number of grain growth steps 
    %in this iteration
    %{
    height_matrix = create_height_matrix(growth_matrix, ...
    init_pore_mapping, max_hight_points);
    num_of_exposed_surface_pores = 0;
    for i = 1:num_of_points_in_subs_vec
        for j = 1:num_of_points_in_subs_vec
            if height_matrix(i, j) == -1
                num_of_exposed_surface_pores = ...
                    num_of_exposed_surface_pores + 1;
            end
        end
    end
    
    exposed_surface_pores_percent = ...
        num_of_exposed_surface_pores / ...
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
    %}
    %{
    num_of_exposed_surface_pore_places = 0;
    for i = 1:num_of_points_in_subs_vec
        for j = 1:num_of_points_in_subs_vec
            if exposed_pores(i, j) == 1
                num_of_exposed_surface_pore_places = ...
                    num_of_exposed_surface_pore_places + 1;
            end
        end
    end
    %}
    num_of_exposed_surface_pore_places = nnz(exposed_pores);
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
        %{
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
        %}
        [idx_i, idx_j] = find(next_it_growth_it_start(:,:,1) == 1);
        for idx = 1:length(idx_i)
            i = idx_i(idx);
            j = idx_j(idx);
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
        %{
        for seed_num = 1:length(active_seeds)
            if active_seeds(seed_num,3) < max_hight_points
                active_seeds(seed_num,3) = active_seeds(seed_num,3)+1;
                it_hight = active_seeds(seed_num,3);
                seed_cen_x = active_seeds(seed_num,1);
                seed_cen_y = active_seeds(seed_num,2);
                max_width = active_seeds(seed_num,4);
                radius_at_hight = min(k*it_hight^(1/cone_exp), max_width);
                for i = max(ceil(seed_cen_x-radius_at_hight), 1):...
                        min(floor(seed_cen_x+radius_at_hight),...
                        num_of_points_in_subs_vec)
                    for j = max(ceil(seed_cen_y-radius_at_hight), 1):...
                            min(floor(seed_cen_y+radius_at_hight),...
                            num_of_points_in_subs_vec)
                        if sqrt((seed_cen_x-i)^2+(seed_cen_y-j)^2) <= ...
                                radius_at_hight
                            growth_matrix(i,j, it_hight) = 1;
                            allowed_places_for_seed(i,j) = 0;
                        end
                    end
                end
            end
        end
        %}
    end
    %breaks the loop if all seeds reached the maximum hight
    %if ~isempty(active_seeds) && min(active_seeds(:,3)) == max_hight_points;
        %break
    %end
    %draws the height profile every 10 iterations
    
    if mod(it, 10) == 0
        %disp('!')
        draw_height_matrix(growth_matrix, ...
            init_pore_mapping, max_hight_points, it);
        place_in_cross_section = floor(num_of_points_in_subs_vec);
        num_horiz_points_in_CS = num_of_points_in_subs_vec - 1;
        adv_horiz_points_in_CS = 1;
        draw_CS_with_depth(place_in_cross_section, ...
            num_horiz_points_in_CS, ...
            adv_horiz_points_in_CS, ...
            max_hight_points, growth_matrix, it)
    end
    
    disp(it)
end
%caluculates porosity and average height
height_matrix = create_height_matrix(growth_matrix, ...
    init_pore_mapping, max_hight_points);

tot_num_of_spaces = 0;
num_of_filled_spaces = 0;
for i = 1:num_of_points_in_subs_vec
    for j = 1:num_of_points_in_subs_vec
        tot_num_of_spaces = tot_num_of_spaces + ...
            height_matrix(i, j);
        for k = 1:max(height_matrix(i, j), 1)
            num_of_filled_spaces = num_of_filled_spaces + ...
                growth_matrix(i,j,k);
        end
    end
end
%tot_num_of_spaces = num_of_points_in_subs_vec...
%    *num_of_points_in_subs_vec*max_hight_points;
mat_size = size(growth_matrix);
avg_height = tot_num_of_spaces / ...
    (mat_size(1) * mat_size(2));
sputtered_porosity = 1 - num_of_filled_spaces/tot_num_of_spaces;
iterations_until_end = it;
disp(sputtered_porosity);
disp(avg_height);
catch ME
    fprintf('Error: %s\n', ME.message);
    diary off;
    exit(1);
end
diary off;

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

%% draws a cross section image of the sputtered layer
function draw_CS_with_depth(place_in_cross_section, num_horiz_points_in_CS, adv_horiz_points_in_CS,...
    max_hight_points, growth_matrix, fig_num)

    cross_section = ones(num_horiz_points_in_CS, max_hight_points) * place_in_cross_section;
    for i = 1:num_horiz_points_in_CS
        for j = 1:max_hight_points
            for depth = 0:place_in_cross_section-1
                if growth_matrix(place_in_cross_section - depth, adv_horiz_points_in_CS + i, j) == 1
                    cross_section(i, j) = depth;
                    break
                end
            end
        end
    end
    cross_section = transpose(cross_section);
    cmap = [gray(place_in_cross_section)/2 + 0.5];

    pixel_size = 0.25; % nm
    x = (0:size(cross_section, 2)-1) * pixel_size;
    y = (0:size(cross_section, 1)-1) * pixel_size;

    fig = figure('Visible', 'off'); % 👈 Headless-compatible
    surf(x, y, cross_section, 'EdgeColor', 'none');
    colormap(cmap);
    view(2);
    title(['iteration number = ' num2str(fig_num)]);
    xlabel('x [nm]');
    ylabel('h [nm]');

    % Save and close
    saveas(fig, sprintf('cross_section_iter_%d.png', fig_num));
    savefig(fig, sprintf('cross_section_iter_%d.fig', fig_num));
    close(fig);
end


%% draws the height matrix in a graph
function draw_height_matrix(growth_matrix, init_pore_mapping, max_hight_points, fig_num)
    height_matrix = create_height_matrix(growth_matrix, init_pore_mapping, max_hight_points);

    pixel_size = 0.25; %nm
    x_spacing = pixel_size;  
    y_spacing = pixel_size;  

    x = (0:size(height_matrix, 2)-1) * x_spacing;
    y = (0:size(height_matrix, 1)-1) * y_spacing;

    pixelated_height_matrix = height_matrix * pixel_size;
    cmap = [0, 0, 0; jet(max_hight_points)];

    fig = figure('Visible', 'off'); % 👈 Headless-compatible figure
    colormap(cmap);
    imagesc(x, y, pixelated_height_matrix);
    colorbar;
    clim(pixel_size * [-1, max_hight_points]);
    axis equal;
    xlabel('[nm]');
    ylabel('[nm]');
    title(['iteration number = ' num2str(fig_num)]);

    % Save and close
    saveas(fig, sprintf('height_matrix_iter_%d.png', fig_num));
    savefig(fig, sprintf('height_matrix_iter_%d.fig', fig_num));
    close(fig);
end

%% creates a mtrix of the height ot the sputterred layer in each point
function height_matrix = create_height_matrix(growth_matrix, init_pore_mapping, max_hight_points)
%mat_size = size(growth_matrix);
%height_matrix = zeros(mat_size(1), mat_size(2));
%{
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
%}
% Collapse growth_matrix along 3rd dim
height_matrix = max(growth_matrix .* reshape(1:max_hight_points, 1, 1, []), [], 3);
% Fill -1 for pores
height_matrix(init_pore_mapping == 0 & height_matrix == 0) = -1;
end


