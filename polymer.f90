module polymer
implicit none
contains

function genpoly(n)
    implicit none
    integer::n,c,m,i,j
    integer,allocatable::seed(:),seed2(:)
    real *8,dimension(n,2)::genpoly ! 0 = north, 1 = east, 2 = south, 3 = west
    real *8::r(n)
    real *8::pi

    pi = 4.0d0*atan(1.0d0)

    call init_random_seed()
    call random_number(r)
    do i=1,n
        genpoly(i,1) = cos(2*pi*r(i))
        genpoly(i,2) = sin(2*pi*r(i))
    enddo

endfunction genpoly

function end2end(chain)
    implicit none
    real *8::chain(:,:)
    real *8::end2end,x,y
    integer::n,i

    x = sum(chain(:,1))
    y = sum(chain(:,2))
    end2end = sqrt(x**2+y**2)

endfunction end2end


subroutine init_random_seed() !Not my code! from user Francesco on Stackoverflow
    INTEGER :: i, n, clock
    INTEGER, DIMENSION(:), ALLOCATABLE :: seed
    CALL RANDOM_SEED(size = n)
    ALLOCATE(seed(n))
    CALL SYSTEM_CLOCK(COUNT=clock)
    seed = clock + 37 * (/ (i - 1, i = 1, n)/)
    CALL RANDOM_SEED(PUT = seed)
    DEALLOCATE(seed)
endsubroutine init_random_seed

endmodule polymer
