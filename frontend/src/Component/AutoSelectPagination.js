import React, { useState } from "react";

import AsyncPaginate from "react-select-async-paginate";


class AutoSelectPagination extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            defaultAdditional: {
                page: 1
            }
        }
    }

    defaultAdditional = {
        page: 1
    };

    loadPageOptions = async (q, prevOptions, { page }) => {
        const { options, hasMore } = await this.props.loadOptions(q, page);
        return {
            options,
            hasMore,

            additional: {
                page: page + 1
            }
        };
    };

    render() {
        const { value, onChange } = this.props;
        return (
            <div className='auto-select'>
                <AsyncPaginate
                    debounceTimeout={600}
                    additional={this.state.defaultAdditional}
                    value={this.props.value}
                    loadOptions={this.loadPageOptions}
                    onChange={onChange}
                />
            </div>
        );
    }
};

export default AutoSelectPagination;