import React, { Component } from 'react';
import { Redirect } from "react-router-dom";
import APIServices from '../../API/apiservices';
import config from '../../Constants/config';
import TopMenuBar from '../../Component/TopMenuBar';
import Pagination from '../../Component/pagination';
import DownloadCSV from '../../Component/DownloadCSV';
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/dist/styles/ag-grid.css';
import 'ag-grid-community/dist/styles/ag-theme-balham.css';
import Spinners from "../../spinneranimation";
import Swal from 'sweetalert2';
import api from '../../API/api'
import cookieStorage from '../../Constants/cookie-storage';
import './AuditLogs.scss';
import '../../ag-grid-table.scss';

const API_URL = config.API_URL;

const apiServices = new APIServices();


class AuditLogs extends Component {
    constructor(props) {
        super(props);
        this.gridApi = null;
        this.state = {
            userName: '',
            selectedEvent: 'Null',
            startDate: '',
            endDate: '',
            page: '',
            device: '',
            event: [],
            eventColumn: [
                { headerName: 'Id', field: 'id' },
                { headerName: 'User Name', field: 'user_name', tooltipField: 'user_name' },
                { headerName: 'Event', field: 'event', tooltipField: 'event' },
                { headerName: 'Page', field: 'page', tooltipField: 'page' },
                { headerName: 'Device Type', field: 'device_type', tooltipField: 'device_type' },
                { headerName: 'Description', field: 'description', tooltipField: 'description', width: 300 },
                { headerName: 'Browser', field: 'browser', tooltipField: 'browser', cellRenderer: (params) => this.browserIcons(params) },
                { headerName: 'Date Time', field: 'date_time', tooltipField: 'date_time' },
                // { headerName: 'Actions', field: '', cellRenderer: (params) => this.actions(params), width: 250 },
            ],
            eventData: [],
            eventData2: [],
            currentPage: '',
            totalPages: '',
            totalRecords: '',
            paginationStart: 1,
            paginationEnd: '',
            paginationSize: '',
            count: 1,
            loading: false,
            searchClick: false,
            url: ''
        };

    }

    componentDidMount() {
        this.getEventList();
        this.setState({

        }, () => this.gotoFirstPage())
    }

    getEventList = () => {
        this.setState({ loading: true })
        api.get(`auditLogs/events`)
            .then((response) => {
                this.setState({ loading: false })
                if (response) {
                    if (response.data.response.length > 0) {
                        this.setState({ event: response.data.response });
                    }
                }
            })
            .catch((err) => {
                this.setState({ loading: false })
            })
    }

    browserIcons = (params) => {
        var browser = params.data.browser
        var element = document.createElement("span");
        var icon = document.createElement("i");

        icon.className = `fa fa-${browser}`
        element.appendChild(icon);
        return element;
    }

    actions(params) {
        var element = document.createElement("span");
        var icon1 = document.createElement("i");
        var icon2 = document.createElement("i");

        icon1.className = 'fa fa-pencil'
        icon1.onclick = () =>
            this.props.history.push({
                pathname: "/editUser",
                state: { data: params.data }
            });

        icon2.className = 'fa fa-trash'
        icon2.onclick = () => {
            Swal.fire({
                title: 'Warning!',
                text: 'Do you want to delete this user?',
                icon: 'warning',
                confirmButtonText: 'Yes',
                showCancelButton: true,
                cancelButtonText: 'No',
                cancelButtonColor: '#ff0000bf'
            }).then((result) => {
                if (result.value) {
                    this.deleteUser(params.data.id)
                }
            })
        }
        element.appendChild(icon1);
        element.appendChild(icon2);
        return element;
    }

    deleteUser = (id) => {
        api.delete(`rest/users/${id}/`)
            .then((res) => {
                Swal.fire({
                    title: 'Success!',
                    text: `User deleted successfully`,
                    icon: 'success',
                    confirmButtonText: 'Ok'
                }).then(() => {
                    this.getUserList();
                })
            })
            .catch((err) => {
                Swal.fire({
                    title: 'Error!',
                    text: `Unable to delete user`,
                    icon: 'error',
                    confirmButtonText: 'Ok'
                })
            })
    }

    handleChange = (e) => {
        this.setState({
            [e.target.id]: e.target.value, count: 1, paginationStart: 1
        });
    }

    callEvents = (e) => {
        this.setState({
            selectedEvent: e.target.value, count: 1, paginationStart: 1
        })
    }

    getFilteredAuditLogs = () => {
        this.setState({ loading: true, eventData: [] })
        const { count, userName, selectedEvent, startDate, endDate, page, device } = this.state;
        const params = { 'page_number': count };

        if (userName) {
            params['username'] = userName
        }
        if (selectedEvent !== 'Null') {
            params['event_id'] = selectedEvent
        }
        if (startDate) {
            params['start_date'] = startDate
        }
        if (endDate) {
            params['end_date'] = endDate
        }
        if (page) {
            params['page_name'] = page
        }
        if (device) {
            params['device_type'] = device
        }

        const serialize = (params) => {
            var str = [];
            for (var p in params)
                if (params.hasOwnProperty(p)) {
                    str.push(encodeURIComponent(p) + "=" + encodeURIComponent(params[p]));
                }
            return str.join("&");
        }

        let url = `auditLogs/logs?${serialize(params)}`
        this.setState({ url })

        api.get(url)
            .then((response) => {
                this.setState({ loading: false })
                if (response) {
                    if (response.data.response) {
                        let data = response.data.response;
                        if (data.events.length > 0) {
                            let eventData = [];
                            data.events.forEach(function (key) {
                                eventData.push({
                                    "id": key.id,
                                    "user_name": key.username,
                                    'description': key.description,
                                    "event": key.event_name,
                                    "page": key.page_name,
                                    "date_time": key.create_date,
                                    'browser': key.browser_name,
                                    'device_type': key.device_type
                                });
                            });
                            if (this.state.searchClick) {
                                if (eventData.length < data.paginationLimit) {
                                    this.setState({ paginationEnd: data.totalRecords }, () => this.setState({ searchClick: false }))
                                } else {
                                    this.setState({ paginationEnd: data.paginationLimit }, () => this.setState({ searchClick: false }))
                                }
                            }
                            this.setState({
                                eventData: eventData,
                                eventData2: eventData,
                                currentPage: data.pageNumber,
                                totalPages: data.totalPages,
                                totalRecords: data.totalRecords,
                                paginationSize: data.paginationLimit,
                            });
                        }
                    }
                }
            })
            .catch((err) => {
                this.setState({ loading: false })
            })
    }

    gotoFirstPage = () => {
        this.setState({
            count: 1,
            paginationStart: 1,
        },
            () => {
                this.getFilteredAuditLogs();
                setTimeout(() => {
                    this.setState({ paginationEnd: this.state.paginationSize })
                }, 1500);
            })
    }

    gotoLastPage = () => {
        const { totalPages, paginationSize, totalRecords } = this.state;
        const startDigit = paginationSize * (totalPages - 1)
        this.setState({
            count: totalPages,
            paginationStart: startDigit + 1,
            paginationEnd: totalRecords
        },
            () => this.getFilteredAuditLogs())
    }

    gotoPreviousPage = () => {
        const { count, currentPage, totalPages, paginationSize, paginationStart, paginationEnd, totalRecords } = this.state;
        const remainder = totalRecords % paginationSize
        const fromLast = currentPage === totalPages
        this.setState({
            count: count - 1,
            paginationStart: paginationStart - paginationSize,
            paginationEnd: paginationEnd - (fromLast ? remainder : paginationSize)
        },
            () => this.getFilteredAuditLogs())
    }

    gotoNextPage = () => {
        const { count, currentPage, totalPages, paginationSize, paginationStart, paginationEnd, totalRecords } = this.state;
        const remainder = totalRecords % paginationSize
        const tolast = currentPage === totalPages - 1
        this.setState({
            count: count + 1,
            paginationStart: paginationStart + paginationSize,
            paginationEnd: paginationEnd + (tolast ? remainder : paginationSize)
        },
            () => this.getFilteredAuditLogs())
    }

    search() {
        this.setState({ searchClick: true }, () => this.getFilteredAuditLogs())
    }

    firstDataRendered = (params) => {
        params.api.sizeColumnsToFit();
    }

    applyColumnLockDown(columnDefs) {
        return columnDefs.map((c) => {
            if (c.children) {
                c.children = this.applyColumnLockDown(c.children);
            }
            return {
                ...c, ...{
                    lockPosition: true,
                    cellClass: 'lock-pinned'
                }
            }
        }
        )
    }

    onGridReady = (params) => {
        if (!params.api) {
            return null;
        }
        this.gridApi = params.api;
    }

    render() {

        setInterval(() => {
            if (this.gridApi) {
                if (this.gridApi.gridPanel.eBodyViewport.clientWidth) {
                    this.gridApi.sizeColumnsToFit();
                }
            }
        }, 1000)

        return (
            <div>
                <TopMenuBar {...this.props} />
                <div className='audit-logs-search'>
                    <div className='add'>
                        <h2>Audit Logs</h2>
                        <DownloadCSV url={`${API_URL}/${this.state.url}`} name={`Audit Logs`} path={`/auditLogs`} page={`Audit Logs Page`} />
                    </div>
                    <div className='module-form'>
                        <div class="form-group">
                            <label for="userName">User Name :</label>
                            <input type="text" class="form-control" id="userName" value={this.state.userName} onChange={(e) => this.handleChange(e)} />
                        </div>

                        <div class="form-group">
                            <label for="events">Events :</label>
                            <select className="form-control cabinselect events-dropdown"
                                onChange={(e) => this.callEvents(e)} id="region">
                                <option selected={true} value="Null">All</option>
                                {this.state.event.map((ele, i) => <option value={ele.id}>{ele.event_name}</option>)}
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="page">Page :</label>
                            <input type="text" class="form-control" id="page" value={this.state.page} onChange={(e) => this.handleChange(e)} />
                        </div>
                        <div class="form-group">
                            <label for="device">Device Type :</label>
                            <input type="text" class="form-control" id="device" value={this.state.device} onChange={(e) => this.handleChange(e)} />
                        </div>
                        <button type="button" className="btn btn-danger" onClick={() => this.search()}>Search</button>
                    </div>
                    <div className='audit-logs-table'>
                        {this.state.loading ?
                            <Spinners /> :
                            <div
                                id='myGrid' className={`ag-theme-balham-dark ag-grid-table`}
                                style={{ height: 'calc(100vh - 350px)' }}
                            >
                                <AgGridReact
                                    onGridReady={this.onGridReady}
                                    onFirstDataRendered={(params) => this.firstDataRendered(params)}
                                    columnDefs={this.applyColumnLockDown(this.state.eventColumn)}
                                    rowData={this.state.eventData}
                                    // onCellClicked={onCellClicked}
                                    rowSelection={`single`}
                                    enableBrowserTooltips={true}
                                >
                                </AgGridReact>
                                <Pagination
                                    paginationStart={this.state.paginationStart}
                                    paginationEnd={this.state.paginationEnd}
                                    totalRecords={this.state.totalRecords}
                                    currentPage={this.state.currentPage}
                                    TotalPages={this.state.totalPages}
                                    gotoFirstPage={() => this.gotoFirstPage()}
                                    gotoLastPage={() => this.gotoLastPage()}
                                    gotoPreviousPage={() => this.gotoPreviousPage()}
                                    gotoNexttPage={() => this.gotoNextPage()}
                                />
                            </div >}
                    </div>
                </div>
            </div>

        );
    }
}

export default AuditLogs;